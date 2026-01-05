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
MEASURED_RUNS = 3

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


def run_with_timing(spark, query_fn, warmup_runs=WARMUP_RUNS, measured_runs=MEASURED_RUNS):
    """Run query multiple times and return timing statistics.
    
    Args:
        spark: SparkSession
        query_fn: Function that takes spark session and returns a DataFrame
        warmup_runs: Number of warmup runs (not measured)
        measured_runs: Number of measured runs
        
    Returns:
        dict with 'avg_ms', 'min_ms', 'max_ms', 'all_ms'
    """
    # Warmup runs - let JIT and caches warm up
    for _ in range(warmup_runs):
        query_fn(spark).collect()
    
    # Measured runs
    times = []
    for _ in range(measured_runs):
        start = time.time()
        result = query_fn(spark).collect()
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    return {
        'avg_ms': sum(times) / len(times),
        'min_ms': min(times),
        'max_ms': max(times),
        'all_ms': times,
        'row_count': len(result),
    }


# Base configuration for PERFILE reader (required for hybrid scan)
base_conf = {
    'spark.rapids.sql.format.parquet.reader.type': 'PERFILE',
    'spark.rapids.sql.reader.chunked': 'false',
    'spark.rapids.memory.pinnedPool.size': '1g',
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


def test_hybrid_scan_multi_column_filter(spark_tmp_path):
    """Test with filter on multiple columns.
    
    This tests the scenario where the filter uses multiple columns, which 
    should exercise the filter column vs payload column split in hybrid scan.
    """
    data_path = spark_tmp_path + '/HYBRID_SHMOO_DATA'
    
    with_cpu_session(
        lambda spark: generate_shmoo_data(spark, data_path, num_rows=NUM_ROWS))
    
    def query_fn(spark):
        return spark.read.parquet(data_path) \
            .filter((col('filter_col') < 10) & (col('payload_1') > 0.5))
    
    # First verify correctness
    assert_gpu_and_cpu_are_equal_collect(query_fn, conf=hybrid_conf)
    
    # Then measure performance
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
# ROW GROUP FILTERING TESTS (using sorted data)
# =============================================================================

@pytest.mark.parametrize('selectivity', [0.01, 0.05, 0.10, 0.25, 0.50], ids=lambda x: f'sel_{int(x*100)}pct')
def test_hybrid_scan_with_row_group_filtering(spark_tmp_path, selectivity):
    """Test hybrid scan with SORTED data that enables row group filtering.
    
    This test creates data sorted by filter_col with many small row groups.
    Each row group will have a narrow min/max range for filter_col, enabling
    the Parquet reader to skip entire row groups based on statistics.
    
    With filter `filter_col < 5`:
    - If data is sorted with 100 row groups, only ~5 row groups are read
    - This should show significant speedup for hybrid scan at low selectivity
    
    NOTE: Using 1M rows to keep file small enough to avoid Spark file splitting.
    The hybrid scan POC does not yet handle split boundaries correctly.
    """
    data_path = spark_tmp_path + '/HYBRID_SHMOO_SORTED'
    num_rows = 10_000_000  # 10M rows for measurable I/O difference
    
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
    
    # Print results with row group info
    print(f"\n{'='*70}")
    print(f"ROW GROUP FILTERING TEST - Sorted Data")
    print(f"{'='*70}")
    print(f"Selectivity: {selectivity*100:.0f}% (filter_col < {threshold})")
    print(f"Total rows: {num_rows:,} | Rows after filter: {hybrid_times['row_count']:,}")
    print(f"{'='*70}")
    print(f"Baseline (Table.readParquet): {baseline_times['avg_ms']:.2f}ms "
          f"(min={baseline_times['min_ms']:.2f}, max={baseline_times['max_ms']:.2f})")
    print(f"Hybrid Scan (PHASE0_POC):     {hybrid_times['avg_ms']:.2f}ms "
          f"(min={hybrid_times['min_ms']:.2f}, max={hybrid_times['max_ms']:.2f})")
    print(f"Speedup: {speedup:.2f}x {'✓ HYBRID FASTER' if speedup > 1 else '✗ BASELINE FASTER'}")
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

