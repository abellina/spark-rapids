/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

package org.apache.spark.sql.rapids.shims.spark321cdh

import org.apache.spark.{SparkConf, SparkEnv, TaskContext}
import org.apache.spark.shuffle._
import org.apache.spark.shuffle.sort.BypassMergeSortShuffleHandle
import org.apache.spark.sql.rapids.{ProxyRapidsShuffleInternalManagerBase, RapidsShuffleInternalManagerBase}
import org.apache.spark.sql.rapids.shims.RapidsShuffleThreadedWriter320

/**
 * A shuffle manager optimized for the RAPIDS Plugin For Apache Spark.
 * @note This is an internal class to obtain access to the private
 *       `ShuffleManager` and `SortShuffleManager` classes.
 */
class RapidsShuffleInternalManager(conf: SparkConf, isDriver: Boolean)
    extends RapidsShuffleInternalManagerBase(conf, isDriver) {

  def getReader[K, C](
      handle: ShuffleHandle,
      startMapIndex: Int,
      endMapIndex: Int,
      startPartition: Int,
      endPartition: Int,
      context: TaskContext,
      metrics: ShuffleReadMetricsReporter): ShuffleReader[K, C] = {
    getReaderInternal(handle, startMapIndex, endMapIndex, startPartition, endPartition, context,
      metrics)
  }

  private lazy val env = SparkEnv.get
  private lazy val blockManager = env.blockManager

  override def getWriter[K, V](
      handle: ShuffleHandle,
      mapId: Long,
      context: TaskContext,
      metricsReporter: ShuffleWriteMetricsReporter): ShuffleWriter[K, V] = {
    handle match {
      case _: BypassMergeSortShuffleHandle[_, _] =>
        new RapidsShuffleThreadedWriter320[K, V](
          blockManager,
          handle.asInstanceOf[BypassMergeSortShuffleHandle[K, V]],
          mapId,
          conf,
          metricsReporter,
          execComponents.get)
      case other =>
        getWriterInternal(handle, mapId, context, metricsReporter)
    }
  }
}


class ProxyRapidsShuffleInternalManager(conf: SparkConf, isDriver: Boolean)
  extends ProxyRapidsShuffleInternalManagerBase(conf, isDriver) with ShuffleManager {

  def getReader[K, C](
    handle: ShuffleHandle,
    startMapIndex: Int,
    endMapIndex: Int,
    startPartition: Int,
    endPartition: Int,
    context: TaskContext,
    metrics: ShuffleReadMetricsReporter
  ): ShuffleReader[K, C] = {
    self.getReader(handle, startMapIndex, endMapIndex, startPartition, endPartition, context,
      metrics)
  }
}