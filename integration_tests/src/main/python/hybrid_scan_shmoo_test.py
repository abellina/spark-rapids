# Copyright (c) 2025, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Hybrid Scan Selectivity Shmoo Tests

These tests measure the performance of hybrid scan (PHASE0_POC) vs the traditional 
Table.readParquet path across different filter selectivity levels.

The goal is to demonstrate that hybrid scan outperforms the baseline when filters 
are highly selective (few rows survive).

See: designs/hybrid_scan/SHMOO_TEST_PLAN.md for test design details.
"""

import pytest
import time
from scipy import stats

from asserts import assert_gpu_and_cpu_are_equal_collect
from spark_session import with_cpu_session, with_gpu_session
from pyspark.sql.functions import col, rand, lit, concat

# Selectivity levels to test (percentage of rows that survive the filter)
# Lower selectivity = more rows filtered out = hybrid scan should win
SELECTIVITY_LEVELS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99]

# Number of rows in test data
NUM_ROWS = 1_000_000

# Number of warmup and measured runs for timing
WARMUP_RUNS = 2
MEASURED_RUNS = 10  # Increased for statistical significance

# Row group size in bytes - smaller = more row groups
# 1MB gives us ~50-100 row groups for 1M rows with this schema
ROW_GROUP_SIZE_BYTES = 1 * 1024 * 1024  # 1 MB


def generate_shmoo_data(spark, data_path, num_rows=NUM_ROWS):
    """Generate test data with uniform distribution for selectivity testing.
    
    Creates column `filter_col` with values 0-99 for easy selectivity control.
    A filter like `filter_col < X` will select approximately X% of rows.
    
    Schema:
    - id: monotonically increasing ID
    - filter_col: value from 0-99 (for selectivity control)
    - payload_1, payload_2, payload_3: random double columns (payload data)
    - payload_str: string column (tests different data types)
    """
    spark.range(num_rows) \
        .withColumn('filter_col', (col('id') % 100).cast('int')) \
        .withColumn('payload_1', rand()) \
        .withColumn('payload_2', rand()) \
        .withColumn('payload_3', rand()) \
        .withColumn('payload_str', concat(lit('row_'), col('id').cast('string'))) \
        .coalesce(4) \
        .write \
        .mode('overwrite') \
        .parquet(data_path)


def generate_shmoo_data_sorted(spark, data_path, num_rows=NUM_ROWS, row_group_size=ROW_GROUP_SIZE_BYTES):
    """Generate test data SORTED by filter_col to enable row group filtering.
    
    By sorting the data by filter_col, each row group will have a narrow min/max range
    for that column. This allows the Parquet reader to skip entire row groups when
    the filter doesn't match the statistics.
    
    Example with 100 row groups:
    - Row group 0: filter_col in [0, 0]
    - Row group 1: filter_col in [1, 1]
    - ...
    - Row group 99: filter_col in [99, 99]
    
    With filter `filter_col < 5`, only 5 row groups need to be read!
    
    Schema:
    - id: original row ID (random order after sort)
    - filter_col: value from 0-99, SORTED for good statistics
    - payload_1, payload_2, payload_3: random double columns (payload data)
    - payload_str: string column (tests different data types)
    """
    # Create data with filter_col values that will cluster when sorted
    # Each filter_col value will have num_rows/100 rows
    spark.range(num_rows) \
        .withColumn('filter_col', (col('id') % 100).cast('int')) \
        .withColumn('payload_1', rand()) \
        .withColumn('payload_2', rand()) \
        .withColumn('payload_3', rand()) \
        .withColumn('payload_str', concat(lit('row_'), col('id').cast('string'))) \
        .orderBy('filter_col') \
        .coalesce(1) \
        .write \
        .mode('overwrite') \
        .option('parquet.block.size', str(row_group_size)) \
        .parquet(data_path)


def generate_shmoo_data_random(spark, data_path, num_rows=NUM_ROWS, row_group_size=ROW_GROUP_SIZE_BYTES):
    """Generate test data with RANDOM filter_col values for AST filtering test.
    
    THIS IS THE KEY TEST FOR HYBRID SCAN!
    
    By having RANDOM filter_col values (0-99) in every row group:
    - Every row group has min=0, max=99 for filter_col
    - Row group statistics CANNOT filter anything
    - Baseline: Reads ALL data, decompresses ALL columns, filters late
    - Hybrid: Applies AST filter EARLY, skips reading payload columns for filtered rows
    
    This is where hybrid scan should show significant improvement!
    
    Schema:
    - filter_col: RANDOM value from 0-99 (uniformly distributed in every row group)
    - payload_1, payload_2, payload_3: random double columns (payload data)
    - payload_str: string column (tests different data types)
    """
    from pyspark.sql.functions import floor
    
    # Create data with RANDOM filter_col values - NOT sorted!
    # This means every row group has min=0, max=99 -> row group pruning won't work
    spark.range(num_rows) \
        .withColumn('filter_col', floor(rand() * 100).cast('int')) \
        .withColumn('payload_1', rand()) \
        .withColumn('payload_2', rand()) \
        .withColumn('payload_3', rand()) \
        .withColumn('payload_str', concat(lit('row_'), col('id').cast('string'))) \
        .drop('id') \
        .coalesce(1) \
        .write \
        .mode('overwrite') \
        .option('parquet.block.size', str(row_group_size)) \
        .parquet(data_path)


def run_with_timing(spark, query_fn, warmup_runs=WARMUP_RUNS, measured_runs=MEASURED_RUNS, 
                    output_path=None):
    """Run query multiple times and return timing statistics.
    
    Args:
        spark: SparkSession
        query_fn: Function that takes spark session and returns a DataFrame
        warmup_runs: Number of warmup runs (not measured)
        measured_runs: Number of measured runs
        output_path: If provided, write to Parquet instead of collecting to driver.
                     This avoids GC overhead for large result sets.
        
    Returns:
        dict with 'avg_ms', 'min_ms', 'max_ms', 'all_ms', 'row_count'
    """
    import tempfile
    import shutil
    
    # Use temp dir if no output path provided
    if output_path is None:
        output_path = tempfile.mkdtemp(prefix='hybrid_scan_timing_')
    
    try:
        # Warmup runs - let JIT and caches warm up
        for _ in range(warmup_runs):
            query_fn(spark).write.mode('overwrite').parquet(output_path)
        
        # Measured runs
        times = []
        for _ in range(measured_runs):
            start = time.time()
            query_fn(spark).write.mode('overwrite').parquet(output_path)
            end = time.time()
            times.append((end - start) * 1000)  # Convert to ms
        
        # Get row count from last run (read back just the count)
        row_count = spark.read.parquet(output_path).count()
        
        return {
            'avg_ms': sum(times) / len(times),
            'min_ms': min(times),
            'max_ms': max(times),
            'all_ms': times,
            'row_count': row_count,
        }
    finally:
        # Clean up temp directory
        shutil.rmtree(output_path, ignore_errors=True)


# Base configuration for PERFILE reader (required for hybrid scan)
base_conf = {
    'spark.rapids.sql.format.parquet.reader.type': 'PERFILE',
    'spark.rapids.sql.reader.chunked': 'false',
    'spark.rapids.memory.pinnedPool.size': '1g',
    'spark.rapids.sql.concurrentGpuTasks': '1',  # Limit GPU concurrency to avoid OOM
}

# Configuration for baseline (Table.readParquet via parquet-mr)
baseline_conf = {
    **base_conf,
    'spark.rapids.sql.format.parquet.hybridScan.mode': 'DISABLED',
}

# Configuration for hybrid scan (PHASE0_POC)
hybrid_conf = {
    **base_conf,
    'spark.rapids.sql.format.parquet.hybridScan.mode': 'PHASE0_POC',
}


@pytest.mark.parametrize('selectivity', SELECTIVITY_LEVELS, ids=lambda x: f'sel_{int(x*100)}pct')
def test_hybrid_scan_selectivity_correctness(spark_tmp_path, selectivity):
    """Verify hybrid scan produces correct results at various selectivity levels.
    
    This test ensures that the hybrid scan path produces the same results as 
    the baseline Table.readParquet path for different filter selectivities.
    """
    data_path = spark_tmp_path + '/HYBRID_SHMOO_DATA'
    
    # Generate data on CPU
    with_cpu_session(
        lambda spark: generate_shmoo_data(spark, data_path, num_rows=100_000))
    
    # Calculate filter threshold (filter_col < X selects ~X% of rows)
    threshold = int(selectivity * 100)
    
    def query_fn(spark):
        return spark.read.parquet(data_path) \
            .filter(col('filter_col') < threshold) \
            .select('id', 'filter_col', 'payload_1', 'payload_2', 'payload_3', 'payload_str')
    
    # Verify correctness: GPU (hybrid) results should match CPU results
    assert_gpu_and_cpu_are_equal_collect(query_fn, conf=hybrid_conf)


@pytest.mark.parametrize('selectivity', SELECTIVITY_LEVELS, ids=lambda x: f'sel_{int(x*100)}pct')
def test_hybrid_scan_selectivity_performance(spark_tmp_path, selectivity):
    """Measure and compare hybrid scan vs baseline performance.
    
    This test measures the execution time of both paths and reports the speedup.
    It does NOT assert any performance requirement - it's a benchmark that outputs
    timing information for analysis.
    
    Expected behavior:
    - Low selectivity (< 25%): Hybrid scan should be faster (reads less data)
    - High selectivity (> 75%): Baseline may be faster (avoids 2-stage overhead)
    """
    data_path = spark_tmp_path + '/HYBRID_SHMOO_DATA'
    
    # Generate data on CPU
    with_cpu_session(
        lambda spark: generate_shmoo_data(spark, data_path, num_rows=NUM_ROWS))
    
    # Calculate filter threshold
    threshold = int(selectivity * 100)
    
    def query_fn(spark):
        return spark.read.parquet(data_path) \
            .filter(col('filter_col') < threshold)
    
    # Run with baseline (Table.readParquet)
    baseline_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=baseline_conf
    )
    
    # Run with hybrid scan (PHASE0_POC)
    hybrid_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=hybrid_conf
    )
    
    # Calculate speedup (>1 means hybrid is faster)
    speedup = baseline_times['avg_ms'] / hybrid_times['avg_ms']
    
    # Print results for analysis
    print(f"\n{'='*60}")
    print(f"Selectivity: {selectivity*100:.0f}% ({threshold}/100 filter threshold)")
    print(f"Rows after filter: {hybrid_times['row_count']:,}")
    print(f"{'='*60}")
    print(f"Baseline (Table.readParquet): {baseline_times['avg_ms']:.2f}ms "
          f"(min={baseline_times['min_ms']:.2f}, max={baseline_times['max_ms']:.2f})")
    print(f"Hybrid Scan (PHASE0_POC):     {hybrid_times['avg_ms']:.2f}ms "
          f"(min={hybrid_times['min_ms']:.2f}, max={hybrid_times['max_ms']:.2f})")
    print(f"Speedup: {speedup:.2f}x {'(Hybrid faster)' if speedup > 1 else '(Baseline faster)'}")
    print(f"{'='*60}\n")


def test_hybrid_scan_no_filter_performance(spark_tmp_path):
    """Measure performance with no filter (100% selectivity).
    
    This is the worst case for hybrid scan - all rows pass, so there's no 
    benefit from the 2-stage approach, only overhead from multiple H2D copies.
    """
    data_path = spark_tmp_path + '/HYBRID_SHMOO_DATA'
    
    with_cpu_session(
        lambda spark: generate_shmoo_data(spark, data_path, num_rows=NUM_ROWS))
    
    def query_fn(spark):
        return spark.read.parquet(data_path)  # No filter
    
    baseline_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=baseline_conf
    )
    
    hybrid_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=hybrid_conf
    )
    
    speedup = baseline_times['avg_ms'] / hybrid_times['avg_ms']
    
    print(f"\n{'='*60}")
    print(f"No Filter (100% selectivity - worst case for hybrid scan)")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_times['avg_ms']:.2f}ms")
    print(f"Hybrid:   {hybrid_times['avg_ms']:.2f}ms")
    print(f"Speedup:  {speedup:.2f}x")
    print(f"{'='*60}\n")


def test_hybrid_scan_high_selectivity_filter(spark_tmp_path):
    """Test with a highly selective filter (best case for hybrid scan).
    
    Filter selects only 1% of rows - this should be where hybrid scan shines
    because it reads much less data from disk.
    """
    data_path = spark_tmp_path + '/HYBRID_SHMOO_DATA'
    
    with_cpu_session(
        lambda spark: generate_shmoo_data(spark, data_path, num_rows=NUM_ROWS))
    
    def query_fn(spark):
        return spark.read.parquet(data_path) \
            .filter(col('filter_col') < 1)  # Only ~1% of rows
    
    baseline_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=baseline_conf
    )
    
    hybrid_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=hybrid_conf
    )
    
    speedup = baseline_times['avg_ms'] / hybrid_times['avg_ms']
    
    print(f"\n{'='*60}")
    print(f"High Selectivity Filter (1% of rows - best case for hybrid)")
    print(f"Rows after filter: {hybrid_times['row_count']:,}")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_times['avg_ms']:.2f}ms")
    print(f"Hybrid:   {hybrid_times['avg_ms']:.2f}ms")
    print(f"Speedup:  {speedup:.2f}x")
    print(f"{'='*60}\n")


def test_hybrid_scan_multi_column_filter_correctness(spark_tmp_path):
    """Verify correctness with filter on multiple columns.
    
    Uses smaller dataset for fast CPU comparison.
    """
    data_path = spark_tmp_path + '/HYBRID_SHMOO_DATA'
    
    # Use small dataset for correctness (CPU comparison)
    with_cpu_session(
        lambda spark: generate_shmoo_data(spark, data_path, num_rows=100_000))
    
    def query_fn(spark):
        return spark.read.parquet(data_path) \
            .filter((col('filter_col') < 10) & (col('payload_1') > 0.5))
    
    assert_gpu_and_cpu_are_equal_collect(query_fn, conf=hybrid_conf)


def test_hybrid_scan_multi_column_filter_performance(spark_tmp_path):
    """Test performance with filter on multiple columns.
    
    This tests the scenario where the filter uses multiple columns, which 
    should exercise the filter column vs payload column split in hybrid scan.
    GPU-only comparison (baseline vs hybrid) for speed.
    """
    data_path = spark_tmp_path + '/HYBRID_SHMOO_DATA'
    
    # Use large dataset for performance testing (GPU only)
    with_cpu_session(
        lambda spark: generate_shmoo_data(spark, data_path, num_rows=NUM_ROWS))
    
    def query_fn(spark):
        return spark.read.parquet(data_path) \
            .filter((col('filter_col') < 10) & (col('payload_1') > 0.5))
    
    # GPU only - compare baseline vs hybrid
    baseline_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=baseline_conf
    )
    
    hybrid_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=hybrid_conf
    )
    
    speedup = baseline_times['avg_ms'] / hybrid_times['avg_ms']
    
    print(f"\n{'='*60}")
    print(f"Multi-Column Filter (filter_col < 10 AND payload_1 > 0.5)")
    print(f"Rows after filter: {hybrid_times['row_count']:,}")
    print(f"{'='*60}")
    print(f"Baseline: {baseline_times['avg_ms']:.2f}ms")
    print(f"Hybrid:   {hybrid_times['avg_ms']:.2f}ms")
    print(f"Speedup:  {speedup:.2f}x")
    print(f"{'='*60}\n")


def test_hybrid_scan_string_filter(spark_tmp_path):
    """Test with string equality filter.
    
    String filters may behave differently due to dictionary encoding in Parquet.
    Hybrid scan can use dictionary-level filtering to skip row groups.
    """
    data_path = spark_tmp_path + '/HYBRID_SHMOO_DATA'
    
    with_cpu_session(
        lambda spark: generate_shmoo_data(spark, data_path, num_rows=100_000))
    
    def query_fn(spark):
        # Filter to a specific string value
        return spark.read.parquet(data_path) \
            .filter(col('payload_str') == 'row_12345')
    
    # Verify correctness
    assert_gpu_and_cpu_are_equal_collect(query_fn, conf=hybrid_conf)


@pytest.mark.parametrize('num_row_groups', [1, 4, 10], ids=lambda x: f'{x}_rg')
def test_hybrid_scan_row_group_count(spark_tmp_path, num_row_groups):
    """Test with varying number of row groups.
    
    More row groups gives hybrid scan more opportunity to skip data via
    statistics-based filtering.
    """
    data_path = spark_tmp_path + '/HYBRID_SHMOO_DATA'
    
    # Coalesce to control number of row groups
    # Note: actual row groups also depend on row group size config
    def generate_with_partitions(spark):
        spark.range(100_000) \
            .withColumn('filter_col', (col('id') % 100).cast('int')) \
            .withColumn('payload_1', rand()) \
            .coalesce(num_row_groups) \
            .write \
            .mode('overwrite') \
            .parquet(data_path)
    
    with_cpu_session(generate_with_partitions)
    
    # Highly selective filter
    def query_fn(spark):
        return spark.read.parquet(data_path) \
            .filter(col('filter_col') < 5)
    
    # Verify correctness
    assert_gpu_and_cpu_are_equal_collect(query_fn, conf=hybrid_conf)
    
    # Measure performance
    hybrid_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn, warmup_runs=1, measured_runs=2),
        conf=hybrid_conf
    )
    
    print(f"\n{num_row_groups} row groups, 5% selectivity: {hybrid_times['avg_ms']:.2f}ms")


# =============================================================================
# AST FILTERING TESTS (using RANDOM data) - THE KEY TEST FOR HYBRID SCAN!
# =============================================================================
# These tests use RANDOM filter_col values so row group statistics CAN'T help.
# This is where hybrid scan's AST filtering should show significant improvement!

@pytest.mark.parametrize('num_rows', [100_000, 1_000_000, 10_000_000, 100_000_000], 
                         ids=lambda x: f'{x//1_000_000}M' if x >= 1_000_000 else f'{x//1_000}K')
@pytest.mark.parametrize('selectivity', [0.01, 0.05, 0.10, 0.25, 0.50], 
                         ids=lambda x: f'sel_{int(x*100)}pct')
def test_hybrid_scan_ast_filtering(spark_tmp_path, selectivity, num_rows):
    """Test hybrid scan with RANDOM data that tests AST filtering.
    
    THIS IS THE KEY TEST FOR HYBRID SCAN PERFORMANCE!
    
    This test creates data with RANDOM filter_col values (0-99) in every row group.
    - Every row group has min=0, max=99 -> row group statistics CAN'T skip anything
    - Baseline: Reads ALL row groups, decompresses ALL columns, filters LATE
    - Hybrid: Applies AST filter EARLY, skips reading payload columns for filtered rows
    
    This is where hybrid scan should show significant speedup!
    """
    data_path = spark_tmp_path + '/HYBRID_SHMOO_RANDOM'
    
    # Generate RANDOM data (NOT sorted - row group stats won't help)
    with_cpu_session(
        lambda spark: generate_shmoo_data_random(spark, data_path, num_rows=num_rows))
    
    # Calculate filter threshold
    threshold = int(selectivity * 100)
    
    def query_fn(spark):
        return spark.read.parquet(data_path) \
            .filter(col('filter_col') < threshold)
    
    # Run with baseline (Table.readParquet)
    baseline_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=baseline_conf
    )
    
    # Run with hybrid scan (PHASE0_POC)
    hybrid_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=hybrid_conf
    )
    
    # Calculate speedup
    speedup = baseline_times['avg_ms'] / hybrid_times['avg_ms']
    
    # Perform Welch's t-test
    t_stat, p_value = stats.ttest_ind(baseline_times['all_ms'], hybrid_times['all_ms'], 
                                       equal_var=False)
    
    # Calculate standard deviations
    import numpy as np
    baseline_std = np.std(baseline_times['all_ms'], ddof=1)
    hybrid_std = np.std(hybrid_times['all_ms'], ddof=1)
    
    # Determine statistical significance
    alpha = 0.05
    is_significant = p_value < alpha
    
    # Helper for readable row count
    def fmt_rows(n):
        if n >= 1_000_000_000:
            return f'{n // 1_000_000_000}B'
        elif n >= 1_000_000:
            return f'{n // 1_000_000}M'
        else:
            return f'{n // 1_000}K'
    
    # Print results
    print(f"\n{'='*70}")
    print(f"AST FILTERING SHMOO (RANDOM DATA) - {fmt_rows(num_rows)} rows, {selectivity*100:.0f}% selectivity")
    print(f"{'='*70}")
    print(f"Filter: filter_col < {threshold}")
    print(f"Input rows: {num_rows:,} | Output rows: {hybrid_times['row_count']:,}")
    print(f"NOTE: Row group stats CAN'T filter - tests AST filtering benefit!")
    print(f"{'='*70}")
    print(f"Baseline (Table.readParquet): {baseline_times['avg_ms']:.2f}ms "
          f"(std={baseline_std:.2f}, min={baseline_times['min_ms']:.2f}, max={baseline_times['max_ms']:.2f})")
    print(f"Hybrid Scan (PHASE0_POC):     {hybrid_times['avg_ms']:.2f}ms "
          f"(std={hybrid_std:.2f}, min={hybrid_times['min_ms']:.2f}, max={hybrid_times['max_ms']:.2f})")
    print(f"Speedup: {speedup:.2f}x {'✓ HYBRID FASTER' if speedup > 1 else '✗ BASELINE FASTER'}")
    print(f"{'='*70}")
    print(f"T-TEST: t={t_stat:.3f}, p={p_value:.4f} {'*** SIGNIFICANT ***' if is_significant else '(not significant)'}")
    print(f"{'='*70}")
    
    # Print CSV-friendly line
    winner = 'Hybrid' if speedup > 1 else 'Baseline'
    sig_marker = '*' if is_significant else ''
    print(f"CSV: AST_RANDOM,{selectivity*100:.0f}%,{fmt_rows(num_rows)},{num_rows},{int(selectivity*num_rows)},"
          f"{baseline_times['avg_ms']:.2f},{baseline_std:.2f},"
          f"{hybrid_times['avg_ms']:.2f},{hybrid_std:.2f},"
          f"{speedup:.2f},{t_stat:.3f},{p_value:.4f},{winner}{sig_marker}")
    print(f"{'='*70}\n")


# =============================================================================
# ROW GROUP FILTERING TESTS (using sorted data) - CONTROL TEST
# =============================================================================
# These tests use SORTED data where row group statistics help BOTH readers equally.
# Expected: similar performance between baseline and hybrid.

# Row counts to test - from small to large
ROW_COUNTS = [100_000, 1_000_000, 10_000_000, 100_000_000]
ROW_COUNTS_LARGE = [1_000_000_000]  # 1B rows - only for low selectivity

def row_count_id(num_rows):
    """Generate readable test ID for row count."""
    if num_rows >= 1_000_000_000:
        return f'{num_rows // 1_000_000_000}B'
    elif num_rows >= 1_000_000:
        return f'{num_rows // 1_000_000}M'
    else:
        return f'{num_rows // 1_000}K'

@pytest.mark.parametrize('num_rows', ROW_COUNTS, ids=row_count_id)
@pytest.mark.parametrize('selectivity', [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 1.00], ids=lambda x: f'sel_{int(x*100)}pct')
def test_hybrid_scan_with_row_group_filtering(spark_tmp_path, selectivity, num_rows):
    """Test hybrid scan with SORTED data that enables row group filtering.
    
    This test creates data sorted by filter_col with many small row groups.
    Each row group will have a narrow min/max range for filter_col, enabling
    the Parquet reader to skip entire row groups based on statistics.
    
    With filter `filter_col < 5`:
    - If data is sorted with 100 row groups, only ~5 row groups are read
    - This should show significant speedup for hybrid scan at low selectivity
    
    GPU-only comparison (baseline vs hybrid) for speed - no CPU runs.
    Parameterized by both selectivity (1-50%) and row count (100K-100M).
    """
    data_path = spark_tmp_path + '/HYBRID_SHMOO_SORTED'
    
    # Generate SORTED data with small row groups (1MB each)
    with_cpu_session(
        lambda spark: generate_shmoo_data_sorted(spark, data_path, num_rows=num_rows))
    
    # Calculate filter threshold
    threshold = int(selectivity * 100)
    
    def query_fn(spark):
        return spark.read.parquet(data_path) \
            .filter(col('filter_col') < threshold)
    
    # Run with baseline (Table.readParquet)
    baseline_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=baseline_conf
    )
    
    # Run with hybrid scan (PHASE0_POC)
    hybrid_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=hybrid_conf
    )
    
    # Calculate speedup
    speedup = baseline_times['avg_ms'] / hybrid_times['avg_ms']
    
    # Perform Welch's t-test (unequal variances t-test)
    # H0: mean(baseline) == mean(hybrid)
    # Ha: mean(baseline) != mean(hybrid) (two-tailed)
    t_stat, p_value = stats.ttest_ind(baseline_times['all_ms'], hybrid_times['all_ms'], 
                                       equal_var=False)  # Welch's t-test
    
    # Calculate standard deviations
    import numpy as np
    baseline_std = np.std(baseline_times['all_ms'], ddof=1)
    hybrid_std = np.std(hybrid_times['all_ms'], ddof=1)
    
    # Determine statistical significance
    alpha = 0.05
    is_significant = p_value < alpha
    
    # Print results with row group info
    print(f"\n{'='*70}")
    print(f"ROW GROUP FILTERING SHMOO - {row_count_id(num_rows)} rows, {selectivity*100:.0f}% selectivity")
    print(f"{'='*70}")
    print(f"Filter: filter_col < {threshold}")
    print(f"Input rows: {num_rows:,} | Output rows: {hybrid_times['row_count']:,}")
    print(f"{'='*70}")
    print(f"Baseline (Table.readParquet): {baseline_times['avg_ms']:.2f}ms "
          f"(std={baseline_std:.2f}, min={baseline_times['min_ms']:.2f}, max={baseline_times['max_ms']:.2f})")
    print(f"Hybrid Scan (PHASE0_POC):     {hybrid_times['avg_ms']:.2f}ms "
          f"(std={hybrid_std:.2f}, min={hybrid_times['min_ms']:.2f}, max={hybrid_times['max_ms']:.2f})")
    print(f"Speedup: {speedup:.2f}x {'✓ HYBRID FASTER' if speedup > 1 else '✗ BASELINE FASTER'}")
    print(f"{'='*70}")
    print(f"T-TEST: t={t_stat:.3f}, p={p_value:.4f} {'*** SIGNIFICANT ***' if is_significant else '(not significant)'}")
    print(f"{'='*70}")
    
    # Print CSV-friendly line for easy data collection
    winner = 'Hybrid' if speedup > 1 else 'Baseline'
    sig_marker = '*' if is_significant else ''
    print(f"CSV: {selectivity*100:.0f}%,{row_count_id(num_rows)},{num_rows},{int(selectivity*num_rows)},"
          f"{baseline_times['avg_ms']:.2f},{baseline_std:.2f},"
          f"{hybrid_times['avg_ms']:.2f},{hybrid_std:.2f},"
          f"{speedup:.2f},{t_stat:.3f},{p_value:.4f},{winner}{sig_marker}")
    print(f"{'='*70}\n")


@pytest.mark.parametrize('num_rows', ROW_COUNTS_LARGE, ids=row_count_id)
@pytest.mark.parametrize('selectivity', [0.01, 0.05, 0.10], ids=lambda x: f'sel_{int(x*100)}pct')
def test_hybrid_scan_1B_low_selectivity(spark_tmp_path, selectivity, num_rows):
    """Test hybrid scan with 1B rows at LOW selectivity only.
    
    1B rows is only tested at 1%, 5%, 10% selectivity to keep output size manageable:
    - 1% of 1B = 10M rows output
    - 5% of 1B = 50M rows output
    - 10% of 1B = 100M rows output
    
    Higher selectivity would produce too much output data and cause memory issues.
    """
    data_path = spark_tmp_path + '/HYBRID_SHMOO_SORTED_1B'
    
    # Generate SORTED data with small row groups (1MB each)
    with_cpu_session(
        lambda spark: generate_shmoo_data_sorted(spark, data_path, num_rows=num_rows))
    
    # Calculate filter threshold
    threshold = int(selectivity * 100)
    
    def query_fn(spark):
        return spark.read.parquet(data_path) \
            .filter(col('filter_col') < threshold)
    
    # Run with baseline (Table.readParquet)
    baseline_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=baseline_conf
    )
    
    # Run with hybrid scan (PHASE0_POC)
    hybrid_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=hybrid_conf
    )
    
    # Calculate speedup
    speedup = baseline_times['avg_ms'] / hybrid_times['avg_ms']
    
    # Perform Welch's t-test (unequal variances t-test)
    t_stat, p_value = stats.ttest_ind(baseline_times['all_ms'], hybrid_times['all_ms'], 
                                       equal_var=False)
    
    # Calculate standard deviations
    import numpy as np
    baseline_std = np.std(baseline_times['all_ms'], ddof=1)
    hybrid_std = np.std(hybrid_times['all_ms'], ddof=1)
    
    # Determine statistical significance
    alpha = 0.05
    is_significant = p_value < alpha
    
    # Print results
    print(f"\n{'='*70}")
    print(f"1B ROW TEST - {row_count_id(num_rows)} rows, {selectivity*100:.0f}% selectivity")
    print(f"{'='*70}")
    print(f"Filter: filter_col < {threshold}")
    print(f"Input rows: {num_rows:,} | Output rows: {hybrid_times['row_count']:,}")
    print(f"{'='*70}")
    print(f"Baseline (Table.readParquet): {baseline_times['avg_ms']:.2f}ms "
          f"(std={baseline_std:.2f}, min={baseline_times['min_ms']:.2f}, max={baseline_times['max_ms']:.2f})")
    print(f"Hybrid Scan (PHASE0_POC):     {hybrid_times['avg_ms']:.2f}ms "
          f"(std={hybrid_std:.2f}, min={hybrid_times['min_ms']:.2f}, max={hybrid_times['max_ms']:.2f})")
    print(f"Speedup: {speedup:.2f}x {'✓ HYBRID FASTER' if speedup > 1 else '✗ BASELINE FASTER'}")
    print(f"{'='*70}")
    print(f"T-TEST: t={t_stat:.3f}, p={p_value:.4f} {'*** SIGNIFICANT ***' if is_significant else '(not significant)'}")
    print(f"{'='*70}")
    
    # Print CSV-friendly line
    winner = 'Hybrid' if speedup > 1 else 'Baseline'
    sig_marker = '*' if is_significant else ''
    print(f"CSV: {selectivity*100:.0f}%,{row_count_id(num_rows)},{num_rows},{int(selectivity*num_rows)},"
          f"{baseline_times['avg_ms']:.2f},{baseline_std:.2f},"
          f"{hybrid_times['avg_ms']:.2f},{hybrid_std:.2f},"
          f"{speedup:.2f},{t_stat:.3f},{p_value:.4f},{winner}{sig_marker}")
    print(f"{'='*70}\n")


def test_hybrid_scan_row_group_filtering_extreme_selectivity(spark_tmp_path):
    """Test with extreme selectivity (0.1%) using sorted data.
    
    This is the ideal case for hybrid scan - only 1 out of ~100 row groups 
    needs to be read.
    """
    data_path = spark_tmp_path + '/HYBRID_SHMOO_SORTED'
    num_rows = 10_000_000  # 10M rows
    
    # Generate SORTED data
    with_cpu_session(
        lambda spark: generate_shmoo_data_sorted(spark, data_path, num_rows=num_rows))
    
    def query_fn(spark):
        # Select only filter_col == 0 (1% of data)
        return spark.read.parquet(data_path) \
            .filter(col('filter_col') == 0)
    
    baseline_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=baseline_conf
    )
    
    hybrid_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=hybrid_conf
    )
    
    speedup = baseline_times['avg_ms'] / hybrid_times['avg_ms']
    
    print(f"\n{'='*70}")
    print(f"EXTREME SELECTIVITY TEST - filter_col == 0 (1%)")
    print(f"{'='*70}")
    print(f"Rows after filter: {hybrid_times['row_count']:,} / {num_rows:,}")
    print(f"Baseline: {baseline_times['avg_ms']:.2f}ms")
    print(f"Hybrid:   {hybrid_times['avg_ms']:.2f}ms")
    print(f"Speedup:  {speedup:.2f}x {'✓ HYBRID FASTER' if speedup > 1 else '✗ BASELINE FASTER'}")
    print(f"{'='*70}\n")


def test_hybrid_scan_row_group_filtering_correctness(spark_tmp_path):
    """Verify correctness of hybrid scan with row group filtering enabled."""
    data_path = spark_tmp_path + '/HYBRID_SHMOO_SORTED'
    
    # Generate sorted data
    with_cpu_session(
        lambda spark: generate_shmoo_data_sorted(spark, data_path, num_rows=100_000))
    
    # Test various filter conditions
    for threshold in [1, 5, 10, 25, 50]:
        def query_fn(spark, t=threshold):
            return spark.read.parquet(data_path) \
                .filter(col('filter_col') < t) \
                .select('id', 'filter_col', 'payload_1', 'payload_2', 'payload_3', 'payload_str')
        
        assert_gpu_and_cpu_are_equal_collect(query_fn, conf=hybrid_conf)
        print(f"✓ Correctness verified for filter_col < {threshold}")


def test_hybrid_scan_row_group_stats_diagnostic(spark_tmp_path):
    """Diagnostic test to verify row group statistics are as expected.
    
    This test generates sorted data and prints information about the 
    resulting Parquet file structure to verify that row groups have 
    non-overlapping filter_col ranges.
    """
    data_path = spark_tmp_path + '/HYBRID_SHMOO_SORTED'
    
    # Generate sorted data with many row groups
    with_cpu_session(
        lambda spark: generate_shmoo_data_sorted(spark, data_path, num_rows=1_000_000,
                                                  row_group_size=512 * 1024))  # 512KB row groups
    
    # Use PyArrow to read file metadata and print row group stats
    try:
        import pyarrow.parquet as pq
        import os
        
        # Find the parquet file(s)
        files = [f for f in os.listdir(data_path) if f.endswith('.parquet')]
        if files:
            file_path = os.path.join(data_path, files[0])
            meta = pq.read_metadata(file_path)
            
            print(f"\n{'='*70}")
            print(f"PARQUET FILE DIAGNOSTIC: {files[0]}")
            print(f"{'='*70}")
            print(f"Number of row groups: {meta.num_row_groups}")
            print(f"Total rows: {meta.num_rows:,}")
            print(f"\nfilter_col statistics per row group:")
            print(f"{'RG':<4} {'Rows':>10} {'Min':>6} {'Max':>6}")
            print(f"{'-'*30}")
            
            for i in range(min(meta.num_row_groups, 20)):  # Show first 20
                rg = meta.row_group(i)
                # filter_col is column index 1 (after 'id')
                col_meta = rg.column(1)
                stats = col_meta.statistics
                if stats:
                    print(f"{i:<4} {rg.num_rows:>10,} {stats.min:>6} {stats.max:>6}")
                else:
                    print(f"{i:<4} {rg.num_rows:>10,} {'N/A':>6} {'N/A':>6}")
            
            if meta.num_row_groups > 20:
                print(f"... ({meta.num_row_groups - 20} more row groups)")
            print(f"{'='*70}\n")
    except ImportError:
        print("PyArrow not available for diagnostic - install with: pip install pyarrow")


# =============================================================================
# HASEEB'S TEST - HIGHLY SELECTIVE AST FILTERING (STRING EQUALITY)
# =============================================================================
# This test replicates Haseeb's data pattern:
# - lookup_value appears ONCE at the START of each row group
# - Row group stats INCLUDE the value (can't skip any RGs)
# - AST filtering skips ~99.9999% of rows within each RG
# - 10 string payload columns (expensive to decompress)
# This is the IDEAL case for hybrid scan!

def generate_haseeb_data(spark, data_path, num_rows=100_000_000, num_rgs=100, num_payload_cols=10,
                         selectivity_pct=0.01):
    """Generate data following Haseeb's pattern for highly selective AST filtering.
    
    Pattern:
    - Each row group has N rows matching lookup_value (at the start of each RG)
    - Row group min/max statistics INCLUDE lookup_value -> can't skip any RGs
    - But AST filtering can skip (100-selectivity)% of rows within each RG
    - String payload columns make skipping very valuable (more cols = more benefit)
    
    Args:
        spark: SparkSession
        data_path: Output path for parquet file
        num_rows: Total number of rows (default 100M)
        num_rgs: Number of row groups (default 100)
        num_payload_cols: Number of string payload columns (default 10)
        selectivity_pct: Percentage of rows that match filter (default 0.01 = 0.01%)
    """
    from pyspark.sql.functions import when, floor, concat_ws, array
    from pyspark.sql.types import StringType
    
    lookup_value = "4500000000"
    rows_per_rg = num_rows // num_rgs
    
    # Calculate how many rows per RG should match the lookup value
    # selectivity_pct is percentage (e.g., 1.0 = 1%, 0.01 = 0.01%)
    matches_per_rg = max(1, int(rows_per_rg * selectivity_pct / 100.0))
    total_matches = matches_per_rg * num_rgs
    actual_selectivity = total_matches / num_rows * 100
    
    # Calculate partitions - ~2.5M rows per partition
    num_partitions = max(1, num_rows // 2_500_000)
    
    print(f"Generating Haseeb test data:")
    print(f"  - {num_rows:,} rows in {num_rgs} row groups")
    print(f"  - {rows_per_rg:,} rows per row group")
    print(f"  - {num_payload_cols} payload columns")
    print(f"  - Lookup value: {lookup_value}")
    print(f"  - Target selectivity: {selectivity_pct}%")
    print(f"  - Matches per RG: {matches_per_rg:,}")
    print(f"  - Expected total matches: {total_matches:,} ({actual_selectivity:.4f}%)")
    print(f"  - Using {num_partitions} partition(s) for data generation")
    
    # Create base DataFrame with partitions upfront to avoid single-node OOM
    # spark.range with numPartitions distributes data generation across executors
    df = spark.range(0, num_rows, 1, num_partitions)
    
    # Create key column:
    # - Position within row group: id % rows_per_rg
    # - If position < matches_per_rg (start of RG), use lookup_value
    # - Otherwise use position as string (unique, won't match)
    position_in_rg = col('id') % rows_per_rg
    df = df.withColumn('key', 
        when(position_in_rg < matches_per_rg, lit(lookup_value))
        .otherwise(position_in_rg.cast('string'))
    )
    
    # Add payload string columns (simulates expensive-to-decompress data)
    for i in range(num_payload_cols):
        df = df.withColumn(f'payload_{i}', 
            concat(lit(f'data_{i}_'), (rand() * 100000).cast('int').cast('string'))
        )
    
    # Remove id column (not in original Haseeb data)
    df = df.drop('id')
    
    # Write with controlled row group size
    row_group_size = max(1024 * 1024, (num_rows * 100) // num_rgs)  # Approximate bytes per RG
    
    df.write \
        .mode('overwrite') \
        .option('parquet.block.size', str(row_group_size)) \
        .parquet(data_path)
    
    print(f"  - Written to {data_path} ({num_partitions} partition(s))")


# Payload column counts to test - more columns = more benefit from skipping
PAYLOAD_COL_COUNTS = [10, 20, 40, 60, 80, 100]

# Selectivity levels to test for Haseeb pattern (percentage of rows matching filter)
HASEEB_SELECTIVITY_LEVELS = [0.01, 1.0, 5.0, 10.0]  # 0.01%, 1%, 5%, 10%


@pytest.mark.parametrize('selectivity_pct', HASEEB_SELECTIVITY_LEVELS, 
                         ids=lambda x: f'sel_{x}pct')
@pytest.mark.parametrize('num_payload_cols', [10, 40, 100], ids=lambda x: f'{x}cols')
@pytest.mark.parametrize('num_rows', [1_000_000], ids=['1M'])
def test_hybrid_scan_haseeb_selectivity(spark_tmp_path, num_rows, num_payload_cols, selectivity_pct):
    """Test hybrid scan varying selectivity with Haseeb pattern.
    
    Tests the sweet spot where hybrid scan wins:
    - Low selectivity (0.01% - 10%) where early filtering saves work
    - Various payload column counts (10, 40, 100)
    
    Selectivity is controlled by placing lookup_value at multiple positions
    at the start of each row group.
    """
    data_path = spark_tmp_path + '/HASEEB_SELECTIVITY'
    num_rgs = 100
    lookup_value = "4500000000"
    num_partitions = max(1, num_rows // 2_500_000)
    
    # Generate test data with specified selectivity
    print(f"\n{'='*80}")
    print(f"HASEEB SELECTIVITY TEST: {num_rows:,} rows, {num_payload_cols} cols, {selectivity_pct}% selectivity")
    print(f"{'='*80}")
    with_cpu_session(
        lambda spark: generate_haseeb_data(spark, data_path, num_rows=num_rows, 
                                           num_rgs=num_rgs, num_payload_cols=num_payload_cols,
                                           selectivity_pct=selectivity_pct))
    
    def query_fn(spark):
        return spark.read.parquet(data_path) \
            .filter(col('key') == lookup_value)
    
    # Run with baseline (Table.readParquet)
    print(f"Running BASELINE (hybridScan=DISABLED)...")
    baseline_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=baseline_conf
    )
    
    # Run with hybrid scan (PHASE0_POC)
    print(f"Running HYBRID SCAN (hybridScan=PHASE0_POC)...")
    hybrid_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=hybrid_conf
    )
    
    # Calculate speedup
    speedup = baseline_times['avg_ms'] / hybrid_times['avg_ms']
    
    # Perform Welch's t-test
    t_stat, p_value = stats.ttest_ind(baseline_times['all_ms'], hybrid_times['all_ms'], 
                                       equal_var=False)
    
    # Calculate standard deviations
    import numpy as np
    baseline_std = np.std(baseline_times['all_ms'], ddof=1)
    hybrid_std = np.std(hybrid_times['all_ms'], ddof=1)
    
    # Statistical significance
    alpha = 0.05
    is_significant = p_value < alpha
    
    # Calculate actual selectivity
    actual_selectivity = hybrid_times['row_count'] / num_rows * 100
    
    # Print results
    print(f"\n{'='*80}")
    print(f"HASEEB SELECTIVITY SHMOO - {selectivity_pct}% target selectivity, {num_payload_cols} cols")
    print(f"{'='*80}")
    print(f"Filter: key == '{lookup_value}'")
    print(f"Input rows: {num_rows:,} | Row groups: {num_rgs} | Payload cols: {num_payload_cols}")
    print(f"Output rows: {hybrid_times['row_count']:,} | Actual selectivity: {actual_selectivity:.4f}%")
    print(f"{'='*80}")
    print(f"Baseline (Table.readParquet): {baseline_times['avg_ms']:.2f}ms "
          f"(std={baseline_std:.2f}, min={baseline_times['min_ms']:.2f}, max={baseline_times['max_ms']:.2f})")
    print(f"Hybrid Scan (PHASE0_POC):     {hybrid_times['avg_ms']:.2f}ms "
          f"(std={hybrid_std:.2f}, min={hybrid_times['min_ms']:.2f}, max={hybrid_times['max_ms']:.2f})")
    print(f"{'='*80}")
    print(f"SPEEDUP: {speedup:.2f}x {'✓ HYBRID FASTER' if speedup > 1 else '✗ BASELINE FASTER'}")
    print(f"T-TEST:  t={t_stat:.3f}, p={p_value:.6f} {'*** SIGNIFICANT ***' if is_significant else '(not significant)'}")
    print(f"{'='*80}")
    
    # CSV output for easy analysis
    winner = 'Hybrid' if speedup > 1 else 'Baseline'
    sig_marker = '*' if is_significant else ''
    print(f"CSV: HASEEB_SEL,{row_count_id(num_rows)},{num_rows},{num_payload_cols},{selectivity_pct},"
          f"{hybrid_times['row_count']},{actual_selectivity:.4f},"
          f"{baseline_times['avg_ms']:.2f},{baseline_std:.2f},"
          f"{hybrid_times['avg_ms']:.2f},{hybrid_std:.2f},"
          f"{speedup:.2f},{t_stat:.3f},{p_value:.6f},{winner}{sig_marker}")
    print(f"{'='*80}\n")


@pytest.mark.parametrize('num_payload_cols', PAYLOAD_COL_COUNTS, ids=lambda x: f'{x}cols')
@pytest.mark.parametrize('num_rows', [1_000_000, 10_000_000, 100_000_000], 
                         ids=['1M', '10M', '100M'])
def test_hybrid_scan_haseeb_payload_cols(spark_tmp_path, num_rows, num_payload_cols):
    """Test hybrid scan varying the number of payload columns.
    
    More payload columns = more data to skip when filtering early.
    This should show increasing benefit for hybrid scan as columns increase.
    
    Tests:
    - 10, 20, 40, 60, 80, 100 payload columns
    - 1M, 10M, 100M rows
    - Highly selective string equality filter (key == lookup_value)
    """
    data_path = spark_tmp_path + '/HASEEB_PAYLOAD_COLS'
    num_rgs = 100
    lookup_value = "4500000000"
    num_partitions = max(1, num_rows // 2_500_000)
    
    # Generate test data with specified payload columns
    print(f"\n{'='*80}")
    print(f"HASEEB PAYLOAD COLS TEST: {num_rows:,} rows, {num_payload_cols} payload columns")
    print(f"{'='*80}")
    with_cpu_session(
        lambda spark: generate_haseeb_data(spark, data_path, num_rows=num_rows, 
                                           num_rgs=num_rgs, num_payload_cols=num_payload_cols))
    
    def query_fn(spark):
        return spark.read.parquet(data_path) \
            .filter(col('key') == lookup_value)
    
    # Run with baseline (Table.readParquet)
    print(f"Running BASELINE (hybridScan=DISABLED)...")
    baseline_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=baseline_conf
    )
    
    # Run with hybrid scan (PHASE0_POC)
    print(f"Running HYBRID SCAN (hybridScan=PHASE0_POC)...")
    hybrid_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=hybrid_conf
    )
    
    # Calculate speedup
    speedup = baseline_times['avg_ms'] / hybrid_times['avg_ms']
    
    # Perform Welch's t-test
    t_stat, p_value = stats.ttest_ind(baseline_times['all_ms'], hybrid_times['all_ms'], 
                                       equal_var=False)
    
    # Calculate standard deviations
    import numpy as np
    baseline_std = np.std(baseline_times['all_ms'], ddof=1)
    hybrid_std = np.std(hybrid_times['all_ms'], ddof=1)
    
    # Statistical significance
    alpha = 0.05
    is_significant = p_value < alpha
    
    # Calculate actual selectivity
    actual_selectivity = hybrid_times['row_count'] / num_rows * 100
    
    # Print results
    print(f"\n{'='*80}")
    print(f"HASEEB PAYLOAD COLUMNS SHMOO - {num_payload_cols} payload columns")
    print(f"{'='*80}")
    print(f"Filter: key == '{lookup_value}'")
    print(f"Input rows: {num_rows:,} | Row groups: {num_rgs} | Payload cols: {num_payload_cols}")
    print(f"Output rows: {hybrid_times['row_count']:,} | Selectivity: {actual_selectivity:.6f}%")
    print(f"{'='*80}")
    print(f"Baseline (Table.readParquet): {baseline_times['avg_ms']:.2f}ms "
          f"(std={baseline_std:.2f}, min={baseline_times['min_ms']:.2f}, max={baseline_times['max_ms']:.2f})")
    print(f"Hybrid Scan (PHASE0_POC):     {hybrid_times['avg_ms']:.2f}ms "
          f"(std={hybrid_std:.2f}, min={hybrid_times['min_ms']:.2f}, max={hybrid_times['max_ms']:.2f})")
    print(f"{'='*80}")
    print(f"SPEEDUP: {speedup:.2f}x {'✓ HYBRID FASTER' if speedup > 1 else '✗ BASELINE FASTER'}")
    print(f"T-TEST:  t={t_stat:.3f}, p={p_value:.6f} {'*** SIGNIFICANT ***' if is_significant else '(not significant)'}")
    print(f"{'='*80}")
    
    # CSV output for easy analysis
    winner = 'Hybrid' if speedup > 1 else 'Baseline'
    sig_marker = '*' if is_significant else ''
    print(f"CSV: HASEEB_COLS,{row_count_id(num_rows)},{num_rows},{num_payload_cols},"
          f"{baseline_times['avg_ms']:.2f},{baseline_std:.2f},"
          f"{hybrid_times['avg_ms']:.2f},{hybrid_std:.2f},"
          f"{speedup:.2f},{t_stat:.3f},{p_value:.6f},{winner}{sig_marker}")
    print(f"{'='*80}\n")


@pytest.mark.parametrize('num_rows', [1_000_000, 10_000_000, 100_000_000], 
                         ids=['1M', '10M', '100M'])
def test_hybrid_scan_haseeb_ast_filter(spark_tmp_path, num_rows):
    """Test hybrid scan with Haseeb's highly selective AST filter pattern.
    
    THIS IS THE DEFINITIVE TEST FOR HYBRID SCAN AST FILTERING!
    
    Data characteristics:
    - String 'key' column with lookup_value at START of each row group
    - Row group statistics INCLUDE lookup_value (min/max covers it)
    - Therefore row group filtering CANNOT skip any row groups
    - Only AST/page-level filtering can skip data
    - 10 string payload columns = expensive to decompress
    
    Expected results:
    - Baseline: Reads ALL data, decompresses ALL columns, filters LATE
    - Hybrid: Applies AST filter EARLY, skips decompressing payloads for 99.99% of rows
    - Speedup should be VERY significant (>10x expected for 100M rows)
    
    This replicates the test from: Haseeb's cudf data generator
    """
    data_path = spark_tmp_path + '/HASEEB_AST_TEST'
    num_rgs = 100
    lookup_value = "4500000000"
    num_partitions = max(1, num_rows // 2_500_000)  # Same as in generate_haseeb_data
    
    # Generate Haseeb's test data
    print(f"\n{'='*80}")
    print(f"HASEEB TEST SETUP: {num_rows:,} rows, {num_rgs} row groups, {num_partitions} partition(s)")
    print(f"{'='*80}")
    with_cpu_session(
        lambda spark: generate_haseeb_data(spark, data_path, num_rows=num_rows, num_rgs=num_rgs))
    
    def query_fn(spark):
        return spark.read.parquet(data_path) \
            .filter(col('key') == lookup_value)
    
    # Run with baseline (Table.readParquet)
    print(f"Running BASELINE (hybridScan=DISABLED)...")
    baseline_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=baseline_conf
    )
    
    # Run with hybrid scan (PHASE0_POC)
    print(f"Running HYBRID SCAN (hybridScan=PHASE0_POC)...")
    hybrid_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=hybrid_conf
    )
    
    # Calculate speedup
    speedup = baseline_times['avg_ms'] / hybrid_times['avg_ms']
    
    # Perform Welch's t-test
    t_stat, p_value = stats.ttest_ind(baseline_times['all_ms'], hybrid_times['all_ms'], 
                                       equal_var=False)
    
    # Calculate standard deviations
    import numpy as np
    baseline_std = np.std(baseline_times['all_ms'], ddof=1)
    hybrid_std = np.std(hybrid_times['all_ms'], ddof=1)
    
    # Statistical significance
    alpha = 0.05
    is_significant = p_value < alpha
    
    # Calculate actual selectivity
    actual_selectivity = hybrid_times['row_count'] / num_rows * 100
    
    # Print results
    print(f"\n{'='*80}")
    print(f"HASEEB'S HYBRID SCAN TEST - HIGHLY SELECTIVE STRING EQUALITY AST FILTER")
    print(f"{'='*80}")
    print(f"Filter: key == '{lookup_value}'")
    print(f"Input rows: {num_rows:,} | Row groups: {num_rgs} | Partitions: {num_partitions}")
    print(f"Output rows: {hybrid_times['row_count']:,} | Selectivity: {actual_selectivity:.6f}%")
    print(f"{'='*80}")
    print(f"KEY INSIGHT: Row group statistics INCLUDE lookup_value (at start of each RG)")
    print(f"            -> Row group filtering CANNOT skip any row groups!")
    print(f"            -> Only AST filtering within RGs provides benefit")
    print(f"{'='*80}")
    print(f"Baseline (Table.readParquet): {baseline_times['avg_ms']:.2f}ms "
          f"(std={baseline_std:.2f}, min={baseline_times['min_ms']:.2f}, max={baseline_times['max_ms']:.2f})")
    print(f"Hybrid Scan (PHASE0_POC):     {hybrid_times['avg_ms']:.2f}ms "
          f"(std={hybrid_std:.2f}, min={hybrid_times['min_ms']:.2f}, max={hybrid_times['max_ms']:.2f})")
    print(f"{'='*80}")
    print(f"SPEEDUP: {speedup:.2f}x {'✓ HYBRID FASTER' if speedup > 1 else '✗ BASELINE FASTER'}")
    print(f"T-TEST:  t={t_stat:.3f}, p={p_value:.6f} {'*** SIGNIFICANT ***' if is_significant else '(not significant)'}")
    print(f"{'='*80}")
    
    # CSV output
    winner = 'Hybrid' if speedup > 1 else 'Baseline'
    sig_marker = '*' if is_significant else ''
    print(f"CSV: HASEEB,{row_count_id(num_rows)},{num_rows},{hybrid_times['row_count']},"
          f"{baseline_times['avg_ms']:.2f},{baseline_std:.2f},"
          f"{hybrid_times['avg_ms']:.2f},{hybrid_std:.2f},"
          f"{speedup:.2f},{t_stat:.3f},{p_value:.6f},{winner}{sig_marker}")
    print(f"{'='*80}\n")


def test_hybrid_scan_haseeb_correctness(spark_tmp_path):
    """Verify correctness of Haseeb's test pattern.
    
    Uses smaller dataset for CPU comparison.
    """
    data_path = spark_tmp_path + '/HASEEB_CORRECTNESS'
    num_rows = 100_000
    num_rgs = 10
    lookup_value = "4500000000"
    
    # Generate small test data
    with_cpu_session(
        lambda spark: generate_haseeb_data(spark, data_path, num_rows=num_rows, num_rgs=num_rgs))
    
    def query_fn(spark):
        return spark.read.parquet(data_path) \
            .filter(col('key') == lookup_value)
    
    # Verify correctness: GPU (hybrid) results should match CPU results
    assert_gpu_and_cpu_are_equal_collect(query_fn, conf=hybrid_conf)
    
    # Verify expected row count
    result_count = with_gpu_session(
        lambda spark: spark.read.parquet(data_path).filter(col('key') == lookup_value).count(),
        conf=hybrid_conf
    )
    
    expected_count = num_rgs  # One match per row group
    print(f"\n{'='*60}")
    print(f"HASEEB CORRECTNESS TEST")
    print(f"{'='*60}")
    print(f"Expected matching rows: {expected_count}")
    print(f"Actual matching rows:   {result_count}")
    print(f"{'='*60}\n")
    
    assert result_count == expected_count, f"Expected {expected_count} rows, got {result_count}"


def test_hybrid_scan_haseeb_range_filter(spark_tmp_path):
    """Test Haseeb's pattern with a range filter (less selective).
    
    This tests string comparison: key < '1000'
    Since key values are position within row group (0, 1, 2, ..., rows_per_rg-1),
    this should match the first ~1000 rows of each row group.
    """
    data_path = spark_tmp_path + '/HASEEB_RANGE'
    num_rows = 10_000_000
    num_rgs = 100
    
    # Generate test data
    with_cpu_session(
        lambda spark: generate_haseeb_data(spark, data_path, num_rows=num_rows, num_rgs=num_rgs))
    
    # String comparison: key < '1000' (matches '0', '1', '10', '100', '1', '11', etc.)
    # Note: String comparison is lexicographic, not numeric!
    filter_value = '1000'
    
    def query_fn(spark):
        return spark.read.parquet(data_path) \
            .filter(col('key') < filter_value)
    
    # Run with baseline
    baseline_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=baseline_conf
    )
    
    # Run with hybrid scan
    hybrid_times = with_gpu_session(
        lambda spark: run_with_timing(spark, query_fn),
        conf=hybrid_conf
    )
    
    speedup = baseline_times['avg_ms'] / hybrid_times['avg_ms']
    selectivity = hybrid_times['row_count'] / num_rows * 100
    
    print(f"\n{'='*70}")
    print(f"HASEEB RANGE FILTER TEST - key < '{filter_value}'")
    print(f"{'='*70}")
    print(f"Input rows: {num_rows:,} | Output rows: {hybrid_times['row_count']:,}")
    print(f"Selectivity: {selectivity:.2f}%")
    print(f"{'='*70}")
    print(f"Baseline: {baseline_times['avg_ms']:.2f}ms")
    print(f"Hybrid:   {hybrid_times['avg_ms']:.2f}ms")
    print(f"Speedup:  {speedup:.2f}x {'✓ HYBRID FASTER' if speedup > 1 else '✗ BASELINE FASTER'}")
    print(f"{'='*70}\n")

