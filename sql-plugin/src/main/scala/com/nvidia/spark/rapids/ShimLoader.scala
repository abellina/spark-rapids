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

import org.apache.spark.{SPARK_BUILD_USER, SPARK_VERSION, SparkConf}
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.VisibleShuffleManager
import org.apache.spark.util.{MutableURLClassLoader, ParentClassLoader}

object ShimLoader extends Logging {

  println(s"GERA_DEBUG: ${this} loaded by ${getClass.getClassLoader}")

  private val shimRootURL = {
    val thisClassFile = getClass.getName.replace(".", "/") + ".class"
    val url = getClass.getClassLoader.getResource(thisClassFile)
    val urlStr = url.toString
    val rootUrlStr = urlStr.substring(0, urlStr.length - thisClassFile.length)
    new URL(rootUrlStr)
  }
  private val shimCommonURL = new URL(s"${shimRootURL.toString}spark3xx-common/")
  private val shimMasks = Seq(
    "spark301",
    "spark302",
    "spark303",
    "spark311",
    "spark312",
    "spark320"
  )

  @volatile private var shimProviderClass: String = _
  @volatile private var sparkShims: SparkShims = _
  @volatile private var shimURL: URL = _
  @volatile private var rapidsJarClassLoader: ClassLoader = _
  // REPL-only logic
  @volatile private var tmpClassLoader: MutableURLClassLoader = _

  def shimId: String = shimProviderClass.split('.').takeRight(2).head

  def getRapidsShuffleManagerClass: String = {
    findShimProvider()
    s"com.nvidia.spark.rapids.$shimId.RapidsShuffleManager"
  }

  def getRapidsShuffleInternal: String = {
    findShimProvider()
    s"org.apache.spark.sql.rapids.shims.$shimId.RapidsShuffleInternalManager"
  }

  private def updateSparkClassLoader(): Unit = {
    val contextClassLoader = Thread.currentThread().getContextClassLoader
    Option(contextClassLoader).collect {
      case mutable: MutableURLClassLoader => mutable
      case replCL if replCL.getClass.getName == "org.apache.spark.repl.ExecutorClassLoader" =>
        val parentLoaderField = replCL.getClass.getDeclaredMethod("parentLoader")
        val parentLoader = parentLoaderField.invoke(replCL).asInstanceOf[ParentClassLoader]
        parentLoader.getParent.asInstanceOf[MutableURLClassLoader]
    }.foreach { mutable =>
      // MutableURLClassloader dedupes for us
      rapidsJarClassLoader = contextClassLoader
      mutable.addURL(shimURL)
      mutable.addURL(shimCommonURL)
    }
  }

  def getShimClassLoader(): ClassLoader = {
    if (shimURL == null) {
      findShimProvider()
    }
    if (rapidsJarClassLoader == null) {
      updateSparkClassLoader()
    }
    if (rapidsJarClassLoader == null) {
      if (tmpClassLoader == null) {
        tmpClassLoader = new MutableURLClassLoader(Array(shimURL, shimCommonURL),
          getClass.getClassLoader)
      }
      tmpClassLoader
    } else {
      rapidsJarClassLoader
    }
  }

  private def detectShimProvider(): String = {
    val sparkVersion = getSparkVersion
    logInfo(s"Loading shim for Spark version: $sparkVersion")
    val shimServiceProviderOpt = shimMasks.flatMap { mask =>
      try {
        val shimURL = new java.net.URL(s"${shimRootURL.toString}$mask/")
        val shimClassLoader = new MutableURLClassLoader(Array(shimURL, shimCommonURL),
          getClass.getClassLoader)
        val shimClass = shimClassLoader
            .loadClass(s"com.nvidia.spark.rapids.shims.$mask.SparkShimServiceProvider")
        Option((instantiateClass(shimClass).asInstanceOf[SparkShimServiceProvider], shimURL))
      } catch {
        case cnf: ClassNotFoundException =>
          logWarning("ignoring " + cnf)
          None
      }
    }.find { case (shimServiceProvider, _) =>
      shimServiceProvider.matchesVersion(sparkVersion)
    }.map { case (inst, url) =>
      shimURL = url
      inst.getClass.getName
    }

    shimServiceProviderOpt.getOrElse {
      throw new IllegalArgumentException(s"Could not find Spark Shim Loader for $sparkVersion")
    }
  }

  private def findShimProvider(): String = {
    // TODO restore support for shim provider override
    if (shimProviderClass == null) {
      shimProviderClass = detectShimProvider()
    }
    shimProviderClass
  }

  def getSparkShims: SparkShims = {
    if (sparkShims == null) {
      sparkShims = newInstanceOf[SparkShimServiceProvider](findShimProvider()).buildShim
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
    instantiateClass(loader.loadClass(className)).asInstanceOf[T]
  }

  def newInternalShuffleManager(conf: SparkConf, isDriver: Boolean): VisibleShuffleManager = {
    getShimClassLoader()
    val shuffleClassName =
      s"org.apache.spark.sql.rapids.shims.${shimId}.RapidsShuffleInternalManager"
    val shuffleClass = tmpClassLoader.loadClass(shuffleClassName)
    shuffleClass.getConstructor(classOf[SparkConf], java.lang.Boolean.TYPE)
        .newInstance(conf, java.lang.Boolean.valueOf(isDriver))
        .asInstanceOf[VisibleShuffleManager]
  }

  // avoid cached constructors
  private def instantiateClass[T](cls: Class[T]): T = {
    logDebug(s"GERA_DEBUG instantiate ${cls.getName} using classloader " + cls.getClassLoader)
    cls.getClassLoader match {
      case m: MutableURLClassLoader => logDebug("GERA_DEBUG urls " + m.getURLs.mkString("\n"))
      case _ =>
    }
    val constructor = cls.getConstructor()
    constructor.newInstance()
  }
}
