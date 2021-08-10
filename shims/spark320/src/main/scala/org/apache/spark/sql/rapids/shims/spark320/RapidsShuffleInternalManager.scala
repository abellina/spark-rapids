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

package org.apache.spark.sql.rapids.shims.spark320

import com.nvidia.spark.rapids.{ShimLoader, ShuffleBufferCatalog}

import org.apache.spark.{ShuffleDependency, SparkConf, TaskContext}
import org.apache.spark.network.buffer.ManagedBuffer
import org.apache.spark.network.shuffle.MergedBlockMeta
import org.apache.spark.shuffle._
import org.apache.spark.sql.rapids.{GpuShuffleBlockResolverBase, RapidsShuffleInternalManagerBase}
import org.apache.spark.storage.ShuffleBlockId

/**
 * A shuffle manager optimized for the RAPIDS Plugin For Apache Spark.
 * @note This is an internal class to obtain access to the private
 *       `ShuffleManager` and `SortShuffleManager` classes.
 */
final class RapidsShuffleInternalManager(conf: SparkConf, isDriver: Boolean)
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

  override protected lazy val resolver = if (shouldFallThroughOnEverything) {
    wrapped.shuffleBlockResolver
  } else {
    new GpuShuffleBlockResolver(wrapped.shuffleBlockResolver, shuffleCatalog)
  }
}

class GpuShuffleBlockResolver(resolver: IndexShuffleBlockResolver, catalog: ShuffleBufferCatalog)
    extends GpuShuffleBlockResolverBase(resolver, catalog) {

  override def getMergedBlockData(
      blockId: ShuffleBlockId,
      dirs: Option[Array[String]]): Seq[ManagedBuffer] = {
    throw new UnsupportedOperationException("TODO after shim is done")
  }

  override def getMergedBlockMeta(
      blockId: ShuffleBlockId,
      dirs: Option[Array[String]]): MergedBlockMeta = {
    throw new UnsupportedOperationException("TODO after shim is done")
  }
}

class ProxyRapidsShuffleInternalManager(conf: SparkConf, isDriver: Boolean)
  extends ShuffleManager {

  val wrapped: ShuffleManager = ShimLoader.newInternalShuffleManager(conf, isDriver)

  override def registerShuffle[K, V, C](
      shuffleId: Int,
      dependency: ShuffleDependency[K, V, C]): ShuffleHandle = {
    wrapped.registerShuffle(shuffleId, dependency)
  }

  override def getWriter[K, V](
      handle: ShuffleHandle,
      mapId: Long,
      context: TaskContext,
      metrics: ShuffleWriteMetricsReporter): ShuffleWriter[K, V] = {
    wrapped.getWriter(handle, mapId, context, metrics)
  }

  override def getReader[K, C](
      handle: ShuffleHandle,
      startMapIndex: Int,
      endMapIndex: Int,
      startPartition: Int,
      endPartition: Int,
      context: TaskContext,
      metrics: ShuffleReadMetricsReporter): ShuffleReader[K, C] = {
    wrapped.getReader(handle, startMapIndex, endMapIndex, startPartition, endPartition, context,
      metrics)
  }

  override def unregisterShuffle(shuffleId: Int): Boolean = {
    wrapped.unregisterShuffle(shuffleId)
  }

  override def shuffleBlockResolver: ShuffleBlockResolver = {
    wrapped.shuffleBlockResolver
  }

  override def stop(): Unit = {
    wrapped.stop()
  }
}

