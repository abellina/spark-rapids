/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids.tool.profiling

import java.io.File
import com.nvidia.spark.rapids.tool.ToolTestUtils
import org.apache.spark.Success
import org.apache.spark.scheduler.{AccumulableInfo, SparkListenerJobStart, SparkListenerTaskEnd, SparkListenerTaskStart, StageInfo, TaskInfo, TaskLocality}
import org.apache.spark.sql.execution.ui.SparkListenerSQLExecutionStart
import org.scalatest.FunSuite
import org.apache.spark.sql.{SparkSession, TrampolineUtil}
import org.apache.spark.sql.rapids.tool.profiling.ApplicationInfo
import org.apache.spark.sql.types._
import org.scalatest.mockito.MockitoSugar

import java.util.Properties

class AnalysisSuite extends FunSuite with MockitoSugar {

  lazy val sparkSession = {
    SparkSession
        .builder()
        .master("local[*]")
        .appName("Rapids Spark Profiling Tool Unit Tests")
        .getOrCreate()
  }

  private val expRoot = ToolTestUtils.getTestResourceFile("ProfilingExpectations")
  private val logDir = ToolTestUtils.getTestResourcePath("spark-events-profiling")
  private val qualLogDir = ToolTestUtils.getTestResourcePath("spark-events-qualification")
  // AutoTuner added a field in SQLTaskAggMetricsProfileResult but it is not among the output
  private val skippedColumnsInSqlAggProfile = Seq("inputBytesReadAvg")

  test("test sqlMetricsAggregation simple") {
    testSqlMetricsAggregation(Array(s"$logDir/rapids_join_eventlog.zstd"),
      "rapids_join_eventlog_sqlmetricsagg_expectation.csv",
      "rapids_join_eventlog_jobandstagemetrics_expectation.csv")
  }

  test("test sqlMetricsAggregation second single app") {
    testSqlMetricsAggregation(Array(s"$logDir/rapids_join_eventlog2.zstd"),
      "rapids_join_eventlog_sqlmetricsagg2_expectation.csv",
      "rapids_join_eventlog_jobandstagemetrics2_expectation.csv")
  }

  test("test sqlMetricsAggregation 2 combined") {
    testSqlMetricsAggregation(
      Array(s"$logDir/rapids_join_eventlog.zstd", s"$logDir/rapids_join_eventlog2.zstd"),
      "rapids_join_eventlog_sqlmetricsaggmulti_expectation.csv",
      "rapids_join_eventlog_jobandstagemetricsmulti_expectation.csv")
  }

  private def testSqlMetricsAggregation(logs: Array[String], expectFile: String,
      expectFileJS: String): Unit = {
    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    assert(apps.size == logs.size)
    val analysis = new Analysis(apps)

    val sqlTaskMetrics = analysis.sqlMetricsAggregation()
    val resultExpectation = new File(expRoot, expectFile)
    import sparkSession.implicits._
    val actualDf = sqlTaskMetrics.toDF.drop(skippedColumnsInSqlAggProfile:_*)
    val dfExpect = ToolTestUtils.readExpectationCSV(sparkSession, resultExpectation.getPath())
    ToolTestUtils.compareDataFrames(actualDf, dfExpect)

    val jobStageMetrics = analysis.jobAndStageMetricsAggregation()
    val resultExpectationJS = new File(expRoot, expectFileJS)
    val actualDfJS = jobStageMetrics.toDF
    val dfExpectJS = ToolTestUtils.readExpectationCSV(sparkSession, resultExpectationJS.getPath())
    ToolTestUtils.compareDataFrames(actualDfJS, dfExpectJS)
  }

  test("test sqlMetrics duration, execute cpu time and potential_problems") {
    val logs = Array(s"$qualLogDir/complex_dec_eventlog.zstd")
    val expectFile = "rapids_duration_and_cpu_expectation.csv"

    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    val analysis = new Analysis(apps)
    // have to call this to set all the fields properly
    analysis.sqlMetricsAggregation()
    import sparkSession.implicits._
    val sqlAggDurCpu = analysis.sqlMetricsAggregationDurationAndCpuTime()
    val resultExpectation = new File(expRoot, expectFile)
    val schema = new StructType()
      .add("appIndex",IntegerType,true)
      .add("appID",StringType,true)
      .add("sqlID",LongType,true)
      .add("sqlDuration",LongType,true)
      .add("containsDataset",BooleanType,true)
      .add("appDuration",LongType,true)
      .add("potentialProbs",StringType,true)
      .add("executorCpuTime",DoubleType,true)
    val actualDf = sqlAggDurCpu.toDF

    val dfExpect = sparkSession.read.option("header", "true").option("nullValue", "-")
      .schema(schema).csv(resultExpectation.getPath())

    ToolTestUtils.compareDataFrames(actualDf, dfExpect)
  }

  test("test shuffleSkewCheck empty") {
    val apps =
      ToolTestUtils.processProfileApps(Array(s"$logDir/rapids_join_eventlog.zstd"), sparkSession)
    assert(apps.size == 1)

    val analysis = new Analysis(apps)
    val shuffleSkewInfo = analysis.shuffleSkewCheck()
    assert(shuffleSkewInfo.isEmpty)
  }

  test("test contains dataset false") {
    val qualLogDir = ToolTestUtils.getTestResourcePath("spark-events-qualification")
    val logs = Array(s"$qualLogDir/nds_q86_test")

    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    val analysis = new Analysis(apps)
    val sqlDurAndCpu = analysis.sqlMetricsAggregationDurationAndCpuTime()
    val containsDs = sqlDurAndCpu.filter(_.containsDataset === true)
    assert(containsDs.isEmpty)
  }

  test("test contains dataset true") {
    val qualLogDir = ToolTestUtils.getTestResourcePath("spark-events-qualification")
    val logs = Array(s"$qualLogDir/dataset_eventlog")

    val apps = ToolTestUtils.processProfileApps(logs, sparkSession)
    val analysis = new Analysis(apps)
    val sqlDurAndCpu = analysis.sqlMetricsAggregationDurationAndCpuTime()
    val containsDs = sqlDurAndCpu.filter(_.containsDataset === true)
    assert(containsDs.size == 1)
  }

  def runMockQuery(ai: ApplicationInfo,
                   metrics: Seq[(String, Long)]): Unit = {
    val ti = new TaskInfo(1, 1, 1, 1, "exec1", "host1", TaskLocality.ANY, false)
    ti.finishTime = 2

    val props = new Properties()
    props.put("spark.sql.execution.id", "1")

    val stageInfo =
      new StageInfo(1, 1, "stage 1", 1, Seq.empty, Seq.empty, "", resourceProfileId = 1)

    ai.processEvent(
      SparkListenerJobStart(1, 0, stageInfo :: Nil, props))

    ai.processEvent(
      SparkListenerSQLExecutionStart(1, "Query 1", "my query", null, null, 0))

    ai.processEvent(
      SparkListenerTaskStart(1, 1, ti))

    TrampolineUtil.setAccumulablesInTaskInfo(
      ti, metrics.zipWithIndex.map { case ((name, update), ix) =>
        AccumulableInfo(ix, Some(name), Some(update), value = None,
          internal = false, countFailedValues = false)
      })

    ai.processEvent(
      SparkListenerTaskEnd(1, 1, "ShuffleMapTask", Success, ti, null,
        TrampolineUtil.makeEmptyTaskMetrics()))
  }

  test("consumes shuffle metrics") {
    val ai = new ApplicationInfo(null, null, 0)
    ai.gpuMode = true

    val metrics = Seq(
      ("rs. shuffle write time", 1000L), ("rs. shuffle write time", 1000L),
      ("rs. shuffle read time", 1500L), ("rs. shuffle read time", 1500L),
      ("rs. shuffle combine time", 250L), ("rs. shuffle combine time", 750L),
      ("rs. shuffle write io time", 250L), ("rs. shuffle write io time", 250L),
      ("rs. serialization time", 250L), ("rs. serialization time", 250L),
      ("rs. deserialization time", 1L), ("rs. deserialization time", 999L))

    runMockQuery(ai, metrics)

    val analysis = new Analysis(ai :: Nil)
    val sqlMetricsAgg :: _ = analysis.sqlMetricsAggregation()

    assertResult(2000)(sqlMetricsAgg.rapidsShuffleWriteTimeNsSum)
    assertResult(1000)(sqlMetricsAgg.rapidsShuffleCombineTimeNsSum)
    assertResult(500)(sqlMetricsAgg.rapidsShuffleWriteIoTimeNsSum)
    assertResult(500)(sqlMetricsAgg.rapidsShuffleSerializationTimeNsSum)

    assertResult(3000)(sqlMetricsAgg.rapidsShuffleReadTimeNsSum)
    assertResult(1000)(sqlMetricsAgg.rapidsShuffleDeserializationTimeNsSum)
  }
}
