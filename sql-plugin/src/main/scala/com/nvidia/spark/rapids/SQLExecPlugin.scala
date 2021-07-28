/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids

import org.apache.spark.internal.Logging
import org.apache.spark.sql.{SparkSession, SparkSessionExtensions}
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.execution.{ColumnarRule, SparkPlan}

class SQLExecPlugin extends (SparkSessionExtensions => Unit) with Logging {
  override def apply(extensions: SparkSessionExtensions): Unit = {
    logRapidsVersionInfo()
    extensions.injectColumnar(shimClassLoaderRule)
    extensions.injectColumnar(columnarOverrides)
    extensions.injectQueryStagePrepRule(queryStagePrepOverrides)
  }

  private def logRapidsVersionInfo() = {
    val pluginProps = RapidsPluginUtils.loadProps(RapidsPluginUtils.PLUGIN_PROPS_FILENAME)
    logInfo(s"RAPIDS Accelerator build: $pluginProps")
    val cudfProps = RapidsPluginUtils.loadProps(RapidsPluginUtils.CUDF_PROPS_FILENAME)
    logInfo(s"cudf build: $cudfProps")
    val pluginVersion = pluginProps.getProperty("version", "UNKNOWN")
    val cudfVersion = cudfProps.getProperty("version", "UNKNOWN")
    logWarning(s"RAPIDS Accelerator $pluginVersion using cudf $cudfVersion." +
        s" To disable GPU support set `${RapidsConf.SQL_ENABLED}` to false")
  }

  private def shimClassLoaderRule(sparkSession: SparkSession): ColumnarRule = {
    // accessing shared state should be fine
    // Base Session State Builder will have inited shared state
    // due to https://scala-lang.org/files/archive/spec/
    // 2.12/06-expressions.html#function-applications
    // function parameter evaluation
    val urls = sparkSession.sharedState.jarClassLoader.getURLs
    logError(s"GERA_DEBUG: Current jar URLs ${urls.mkString("\n")}")
    val shimURL = ShimLoader.getShimURL
    logError(s"GERA_DEBUG adding Shim URL $shimURL")
    sparkSession.sharedState.jarClassLoader.addURL(shimURL)
    ShimLoader.getSparkShims // IMPORTANT: force Shim load before rule injection
    new ColumnarRule // identity
  }

  private def columnarOverrides(sparkSession: SparkSession): ColumnarRule = {
    sparkSession.sharedState.jarClassLoader
        .loadClass("com.nvidia.spark.rapids.ColumnarOverrideRules")
        .newInstance()
        .asInstanceOf[ColumnarRule]
  }

  private def queryStagePrepOverrides(sparkSession: SparkSession): Rule[SparkPlan] = {
    sparkSession.sharedState.jarClassLoader
        .loadClass("com.nvidia.spark.rapids.GpuQueryStagePrepOverrides")
        .newInstance()
        .asInstanceOf[Rule[SparkPlan]]
  }
}