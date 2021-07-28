/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.shims.spark311

import com.nvidia.spark.rapids.{SparkShims, SparkShimVersion}
import com.nvidia.spark.rapids.shims.spark311.SparkShimServiceProvider.shimClassName

import org.apache.spark.sql.SparkSession
import org.apache.spark.util.MutableURLClassLoader

object SparkShimServiceProvider {
  val VERSION = SparkShimVersion(3, 1, 1)
  val VERSIONNAMES = Seq(s"$VERSION")
  val shimClassName = "com.nvidia.spark.rapids.shims.spark311.Spark311Shims"
}

class SparkShimServiceProvider extends com.nvidia.spark.rapids.SparkShimServiceProvider {

  def matchesVersion(version: String): Boolean = {
    SparkShimServiceProvider.VERSIONNAMES.contains(version)
  }

  def buildShim: SparkShims = {
    // TODO hack
    val sparkSession = SparkSession.getActiveSession
    println("Spark Session " + sparkSession)
    sparkSession.map { sparkSession =>
      val classLoader = sparkSession.sharedState.jarClassLoader
      println("Using session classloader: " + classLoader)
      println("  URLs " + classLoader.asInstanceOf[MutableURLClassLoader].getURLs.mkString("\n"))
      classLoader
    }.getOrElse {
      // this for non-Spark apps like RapidsConf that don't init Spark sessions
      val classLoader = classOf[SparkShims].getClassLoader
      println("Using caller's classloader: " + classLoader)
      classLoader
    }.loadClass(shimClassName).newInstance().asInstanceOf[SparkShims]
  }
}
