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
import org.apache.spark.sql.execution.ColumnarRule

class SQLExecPlugin extends (SparkSessionExtensions => Unit) with Logging {
  override def apply(extensions: SparkSessionExtensions): Unit = {
    val pluginProps = RapidsPluginUtils.loadProps(RapidsPluginUtils.PLUGIN_PROPS_FILENAME)
    logInfo(s"RAPIDS Accelerator build: $pluginProps")
    val cudfProps = RapidsPluginUtils.loadProps(RapidsPluginUtils.CUDF_PROPS_FILENAME)
    logInfo(s"cudf build: $cudfProps")
    val pluginVersion = pluginProps.getProperty("version", "UNKNOWN")
    val cudfVersion = cudfProps.getProperty("version", "UNKNOWN")
    logWarning(s"RAPIDS Accelerator $pluginVersion using cudf $cudfVersion." +
        s" To disable GPU support set `${RapidsConf.SQL_ENABLED}` to false")
    val columnarRules: SparkSession => ColumnarRule = { sparkSession =>
      val urls = sparkSession.sharedState.jarClassLoader.getURLs
      logError(s"GERA_DEBUG: Current jar URLs $urls")
      val shimURL = ShimLoader.getShimURL
      logError(s"GERA_DEBUG adding Shim URL $shimURL")
      sparkSession.sharedState.jarClassLoader.addURL(shimURL)
      ColumnarOverrideRules()
    }

    extensions.injectColumnar(columnarRules)
    extensions.injectQueryStagePrepRule(_ => GpuQueryStagePrepOverrides())
  }
}