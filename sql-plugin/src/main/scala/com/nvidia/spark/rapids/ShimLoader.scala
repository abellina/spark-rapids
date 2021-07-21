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

package com.nvidia.spark.rapids

import scala.io.Source
import scala.util.Try

import com.sun.istack.internal.tools.ParallelWorldClassLoader

import org.apache.spark.{SPARK_BUILD_USER, SPARK_VERSION}
import org.apache.spark.internal.Logging

object ShimLoader extends Logging {
  private var shimProviderClass: String = null
  private var sparkShims: SparkShims = null

  private def loadShimProviders(cl: ClassLoader): Iterator[SparkShimServiceProvider] = {
    val serviceResourceListFile = "META-INF/services/" + classOf[SparkShimServiceProvider].getName
    logError("GERA_DEBUG: loading " + serviceResourceListFile)
    Option(cl.getResourceAsStream(serviceResourceListFile))
        .map(is => Source.fromInputStream(is).getLines())
        .getOrElse {
          logError("GERA_DEBUG: no service providers found")
          Iterator.empty
        }
        .flatMap(provideClass => Try(
          cl.loadClass(provideClass).newInstance().asInstanceOf[SparkShimServiceProvider]
        ).toOption)
  }

  private def detectShimProvider(): SparkShimServiceProvider = {
    val shimClassloaders = Seq(
      "spark301",
      "spark302",
      "spark303",
      "spark311",
      "spark312"
    ).map(prefix => new ParallelWorldClassLoader(getClass.getClassLoader, s"$prefix/"))

    val sparkVersion = getSparkVersion
    logInfo(s"Loading shim for Spark version: $sparkVersion")
    // TODO current fat jar has multiple copies due to being merged from other fat jars
    val shimLoader = shimClassloaders.toIterator
        .flatMap(cl =>
          loadShimProviders(cl)
        ).find(_.matchesVersion(sparkVersion))

    shimLoader.getOrElse {
      throw new IllegalArgumentException(s"Could not find Spark Shim Loader for $sparkVersion")
    }
  }

  private def findShimProvider(): SparkShimServiceProvider = {
    if (shimProviderClass == null) {
      detectShimProvider()
    } else {
      logWarning(s"Overriding Spark shims provider to $shimProviderClass. " +
          "This may be an untested configuration!")
      val providerClass = Class.forName(shimProviderClass)
      val constructor = providerClass.getConstructor()
      constructor.newInstance().asInstanceOf[SparkShimServiceProvider]
    }
  }

  def getSparkShims: SparkShims = {
    if (sparkShims == null) {
      val provider = findShimProvider()
      sparkShims = provider.buildShim
    }
    sparkShims
  }

  def getSparkVersion: String = {
    // hack for databricks, try to find something more reliable?
    if (SPARK_BUILD_USER.equals("Databricks")) {
        SPARK_VERSION + "-databricks"
    } else {
      SPARK_VERSION
    }
  }

  def setSparkShimProviderClass(classname: String): Unit = {
    shimProviderClass = classname
  }
}
