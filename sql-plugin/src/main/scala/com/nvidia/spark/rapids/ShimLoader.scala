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
    onExecutor = true
    this
  }

  def getExecutorContextClassloader(): Option[ClassLoader] = {
    val pluginClassLoaderURL = ShimLoader.getShimURL()
    updateExecutorClassLoader(pluginClassLoaderURL)
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
    }.map { mutable =>
      // MutableURLClassloader dedupes for us
      mutable.addURL(pluginClassLoaderURL)
      mutable
    }
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
      jarClassLoader.addURL(shimRootURL)
      rapidsJarClassLoader = jarClassLoader
      getShimURL()
//      updateExecutorClassLoader(getShimURL())
    }
    rapidsJarClassLoader
//    SparkSession.getActiveSession.map(_.sharedState.jarClassLoader)
//        .orElse(getExecutorContextClassloader())
//        .getOrElse(getClass.getClassLoader)
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
    shimMasks.flatMap { mask =>
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
        rapidsJarClassLoader = Thread.currentThread().getContextClassLoader
        rapidsJarClassLoader.loadClass(inst.getClass.getName)
            .newInstance().asInstanceOf[SparkShimServiceProvider]
      } else {
        rapidsJarClassLoader = inst.getClass.getClassLoader
        inst
      }
    }.getOrElse {
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

  def newInstanceOf[T](className: String): T = {
    val loader = getShimClassLoader()
    logDebug(s"Loading $className using $loader with the parent loader ${loader.getParent}")
    loader.loadClass(className).newInstance().asInstanceOf[T]
  }
}
