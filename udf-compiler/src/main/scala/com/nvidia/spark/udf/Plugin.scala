/*
 * Copyright (c) 2019-2021, NVIDIA CORPORATION.
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

package com.nvidia.spark.udf

import com.nvidia.spark.rapids.ShimLoader

import org.apache.spark.internal.Logging
import org.apache.spark.sql.{SparkSession, SparkSessionExtensions}
import org.apache.spark.sql.catalyst.plans.logical.LogicalPlan
import org.apache.spark.sql.catalyst.rules.Rule

class Plugin extends (SparkSessionExtensions => Unit) with Logging {
  override def apply(extensions: SparkSessionExtensions): Unit = {
    // TODO move this into ShimLoader-loade classes
//    logWarning("Installing rapids UDF compiler extensions to Spark. The compiler is disabled" +
//        s" by default. To enable it, set `${RapidsConf.UDF_COMPILER_ENABLED}` to true")

    extensions.injectResolutionRule(logicalPlanRules)
  }

  def logicalPlanRules(sparkSession: SparkSession): Rule[LogicalPlan] = {
    ShimLoader.newInstanceOf("com.nvidia.spark.udf.LogicalPlanRules")
  }
}
