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

import java.net.URL

import org.apache.spark.{SPARK_BUILD_USER, SPARK_VERSION}
import org.apache.spark.api.plugin.ExecutorPlugin
import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.{MutableURLClassLoader, ParentClassLoader}

object ShimLoader extends Logging {
  private var shimProvider: SparkShimServiceProvider = null
  private var shimProviderClass: String = null
  private var sparkShims: SparkShims = null
  private var shimURL: URL = null
  val shimMasks = Seq(
    "spark301",
    "spark302",
    "spark303",
    "spark311",
    "spark312",
    "spark320"
  )

  def getRapidsShuffleManagerClass: String = {
    val provider = findShimProvider()
    s"${provider.getClass.getPackage.getName}.RapidsShuffleManager"
  }

  def getExecutorContextClassloader(): Option[ClassLoader] = {
    val pluginClassLoaderURL = ShimLoader.getShimURL()
    val contextClassLoader = Thread.currentThread().getContextClassLoader
    Option(contextClassLoader).collect {
      case mutable: MutableURLClassLoader => mutable
      case replCL if replCL.getClass.getName == "org.apache.spark.repl.ExecutorClassLoader" =>
        val parentLoaderField = replCL.getClass.getDeclaredMethod("parentLoader")
        val parentLoader = parentLoaderField.invoke(replCL).asInstanceOf[ParentClassLoader]
        parentLoader.getParent.asInstanceOf[MutableURLClassLoader]
    }.map { mutable =>
      // MutableURLClassloader dedupes for us
      mutable.addURL(pluginClassLoaderURL)
      mutable
    }
  }

  def shimClassLoader(): ClassLoader = {
    SparkSession.getActiveSession.map(_.sharedState.jarClassLoader)
        .orElse(getExecutorContextClassloader())
        .getOrElse(getClass.getClassLoader)
  }

  def getShimURL(): URL = {
    if (shimURL == null) {
      // base case: we rely on provider classes not being in parallel world
      val providerClassLoader = getClass.getClassLoader
      val rsrcURL = providerClassLoader.getResource(
        findShimProvider().getClass.getName.replace(".", "/") + ".class"
      )
      // jar:file:/path/rapids-4-spark_2.12-<version>.jar
      // !/com/nvidia/spark/rapids/shims/spark301/Class.class
      val urlStr = rsrcURL.toString
      val shimRootUrlStr = urlStr.substring(0, urlStr.indexOf("!") + 1) +
          "/" + shimProvider.getClass.getPackage.getName.split('.').last +
          "/" // IMPORTANT: trailing slash for the loadClass contract
      shimURL = new URL(shimRootUrlStr)
    }
    shimURL
  }

  private def detectShimProvider(): SparkShimServiceProvider = {
    val sparkVersion = getSparkVersion
    logInfo(s"Loading shim for Spark version: $sparkVersion")
    val shimLoader = shimMasks.flatMap { mask =>
      try {
         // base case: we rely on provider classes not being in parallel world
         val shimClass = getClass.getClassLoader
            .loadClass(s"com.nvidia.spark.rapids.shims.$mask.SparkShimServiceProvider")
        Option(shimClass.newInstance().asInstanceOf[SparkShimServiceProvider])
      } catch {
        case cnf: ClassNotFoundException =>
          logWarning("ignoring " + cnf)
          None
      }
    }.find(_.matchesVersion(sparkVersion))

    shimLoader.getOrElse {
      throw new IllegalArgumentException(s"Could not find Spark Shim Loader for $sparkVersion")
    }
  }

  private def findShimProvider(): SparkShimServiceProvider = {
    Option(shimProvider).getOrElse {
      if (shimProviderClass == null) {
        shimProvider = detectShimProvider()
      } else {
        logWarning(s"Overriding Spark shims provider to $shimProviderClass. " +
            "This may be an untested configuration!")
        val providerClass = Class.forName(shimProviderClass)
        val constructor = providerClass.getConstructor()
        shimProvider = constructor.newInstance().asInstanceOf[SparkShimServiceProvider]
      }
      shimProvider
    }
  }

  def getSparkShims: SparkShims = {
    if (sparkShims == null) {
      val provider = findShimProvider()
      sparkShims = provider.buildShim

      // TODO related to https://stackoverflow.com/questions/28186607/java-lang-classcastexception
      //  -using-lambda-expressions-in-spark-job-on-remote-ser/28367602#28367602
      // is there a better way to propagate the classloader to serde
//      Thread.currentThread().setContextClassLoader(sparkShims.getClass.getClassLoader)
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

  def executorPlugin(): ExecutorPlugin = {
    shimClassLoader()
        .loadClass(getClass.getPackage.getName + ".RapidsExecutorPlugin")
        .newInstance()
        .asInstanceOf[ExecutorPlugin]
  }
}
