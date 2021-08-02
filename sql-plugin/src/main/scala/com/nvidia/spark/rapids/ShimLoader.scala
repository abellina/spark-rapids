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

import org.apache.spark.{SPARK_BUILD_USER, SPARK_VERSION, SparkEnv}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.SparkSession
import org.apache.spark.util.{ChildFirstURLClassLoader, MutableURLClassLoader, ParentClassLoader}

object ShimLoader extends Logging {

  val shimRootURL = {
    val thisClassFile = getClass.getName.replace(".", "/") + ".class"
    val url = getClass.getClassLoader.getResource(thisClassFile)
    val urlStr = url.toString
    val rootUrlStr = urlStr.substring(0, urlStr.length - thisClassFile.length)
    new URL(rootUrlStr)
  }

  private var onExecutor: Boolean = _
  private var shimProvider: SparkShimServiceProvider = _
  private var shimProviderClass: String = _
  private var sparkShims: SparkShims = _
  private var shimURL: URL = _
  private val shimMasks = Seq(
    "spark301",
    "spark302",
    "spark303",
    "spark311",
    "spark312",
    "spark320"
  )

  private var rapidsJarClassLoader: ClassLoader = _

  def getRapidsShuffleManagerClass: String = {
    val provider = findShimProvider()
    s"${provider.getClass.getPackage.getName}.RapidsShuffleManager"
  }

  def forExecutor() = {
    if (SparkEnv.get.executorId == "driver") {
      // TODO smoother integration
      // in the local mode we have already inited the ShimProvider
      // just update the executor classloader so it can deserialize
      updateExecutorClassLoader(shimURL)
    } else {
      onExecutor = true
    }
    this
  }

  private def updateExecutorClassLoader(pluginClassLoaderURL: URL) = {
    require(pluginClassLoaderURL != null, "Couldn't locate shim provider?")
    val contextClassLoader = Thread.currentThread().getContextClassLoader

    Option(contextClassLoader).collect {
      case mutable: MutableURLClassLoader => mutable
      case replCL if replCL.getClass.getName == "org.apache.spark.repl.ExecutorClassLoader" =>
        val parentLoaderField = replCL.getClass.getDeclaredMethod("parentLoader")
        val parentLoader = parentLoaderField.invoke(replCL).asInstanceOf[ParentClassLoader]
        parentLoader.getParent.asInstanceOf[MutableURLClassLoader]
    }.foreach { mutable =>
      // MutableURLClassloader dedupes for us
      mutable.addURL(pluginClassLoaderURL)
    }
    rapidsJarClassLoader = Thread.currentThread().getContextClassLoader
  }

  def getShimClassLoader(): ClassLoader = {
    if (rapidsJarClassLoader == null) {
      // TODO initially it ends up delegating to the parent,
      // in the future we will hide all common classes from parent under a dedicated directory
      //
      val jarClassLoader = new MutableURLClassLoader(Array.empty,
        Option(Thread.currentThread().getContextClassLoader)
            .getOrElse(classOf[SparkSession].getClassLoader)
      )
      rapidsJarClassLoader = jarClassLoader
      getShimURL()
//      updateExecutorClassLoader(getShimURL())
    }
    rapidsJarClassLoader
  }

  def getShimURL(): URL = {
    if (shimURL == null) {
      findShimProvider()
    }
    shimURL
  }

  private def detectShimProvider(): SparkShimServiceProvider = {
    val sparkVersion = getSparkVersion
    logInfo(s"Loading shim for Spark version: $sparkVersion")
    val shimServiceProviderOpt = shimMasks.flatMap { mask =>
      try {
        val shimURL = new java.net.URL(s"${shimRootURL.toString}$mask/")
        val shimClassLoader = new ChildFirstURLClassLoader(Array(shimURL),
          rapidsJarClassLoader)
        val shimClass = shimClassLoader
            .loadClass(s"com.nvidia.spark.rapids.shims.$mask.SparkShimServiceProvider")
        Option((shimClass.newInstance().asInstanceOf[SparkShimServiceProvider], shimURL))
      } catch {
        case cnf: ClassNotFoundException =>
          logWarning("ignoring " + cnf)
          None
      }
    }.find { case (shimServiceProvider, _) =>
      shimServiceProvider.matchesVersion(sparkVersion)
    }.map { case (inst, url) =>
      shimURL = url
      if (onExecutor) {
        updateExecutorClassLoader(shimURL)
      } else {
        // TODO cleanup
        rapidsJarClassLoader
          .asInstanceOf[MutableURLClassLoader]
          .addURL(shimURL)
      }
      rapidsJarClassLoader.loadClass(inst.getClass.getName)
            .newInstance().asInstanceOf[SparkShimServiceProvider]
    }

    shimServiceProviderOpt.getOrElse {
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
        shimProvider = newInstanceOf(shimProviderClass)
      }
      shimProvider
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

  def newInstanceOf[T](className: String): T = {
    val loader = getShimClassLoader()
    logDebug(s"Loading $className using $loader with the parent loader ${loader.getParent}")
    loader.loadClass(className).newInstance().asInstanceOf[T]
  }
}
