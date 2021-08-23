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
package org.apache.spark.sql.rapids.shims.spark320

import com.nvidia.spark.rapids.ShimLoader
import org.apache.spark.{Dependency, MapOutputTrackerMaster, Partition, ShuffleDependency, SparkContext, SparkEnv, TaskContext}
import org.apache.spark.shuffle.{ShuffleHandle, ShuffleManager, ShuffleReadMetricsReporter, ShuffleReader}
import org.apache.spark.sql.execution.{CoalescedPartitionSpec, PartialMapperPartitionSpec, PartialReducerPartitionSpec}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLShuffleReadMetricsReporter}
import org.apache.spark.sql.rapids.ShuffleManagerShimBase
import org.apache.spark.sql.rapids.execution.ShuffledBatchRDDPartition
import org.apache.spark.sql.vectorized.ColumnarBatch

class ShuffleManagerShim extends ShuffleManagerShimBase {

  override def getPreferredLocations(
      dependency: ShuffleDependency[Int, ColumnarBatch, ColumnarBatch],
      partition: Partition): Seq[String] = {
    val tracker = SparkEnv.get.mapOutputTracker.asInstanceOf[MapOutputTrackerMaster]
    partition.asInstanceOf[ShuffledBatchRDDPartition].spec match {
      case CoalescedPartitionSpec(startReducerIndex, endReducerIndex, dataSize) =>
        // TODO order by partition size.
        startReducerIndex.until(endReducerIndex).flatMap { reducerIndex =>
          tracker.getPreferredLocationsForShuffle(dependency, reducerIndex)
        }

      case prps: PartialReducerPartitionSpec =>
        tracker.getMapLocation(dependency, prps.startMapIndex, prps.endMapIndex)

      case PartialMapperPartitionSpec(mapIndex, _, _) =>
        tracker.getMapLocation(dependency, mapIndex, mapIndex + 1)
    }
  }

  override def getReader[K, C](
      shuffleManager: ShuffleManager,
      handle: ShuffleHandle,
      startMapIndex: Int,
      endMapIndex: Int,
      startPartition: Int,
      endPartition: Int,
      context: TaskContext,
      metrics: ShuffleReadMetricsReporter): ShuffleReader[K, C] = {
    shuffleManager.getReader(
      handle, startMapIndex, endMapIndex, startPartition, endPartition, context, metrics)
  }

  override def getReaderAndPartitionSizeForSpec[K, C](
      context: TaskContext,
      dependency: ShuffleDependency[Int, ColumnarBatch, ColumnarBatch],
      shuffledBatchRDDPartition:  ShuffledBatchRDDPartition,
      sqlMetricsReporter: SQLShuffleReadMetricsReporter): (ShuffleReader[K, C], Long) = {
    val shim = ShimLoader.getSparkShims
    shuffledBatchRDDPartition.spec match {
      case CoalescedPartitionSpec(startReducerIndex, endReducerIndex, dataSize) =>
        val reader = SparkEnv.get.shuffleManager.getReader[K, C](
          dependency.shuffleHandle,
          startReducerIndex,
          endReducerIndex,
          context,
          sqlMetricsReporter)
        val blocksByAddress = shim.getMapSizesByExecutorId(
          dependency.shuffleHandle.shuffleId, 0, Int.MaxValue, startReducerIndex, endReducerIndex)
        val partitionSize = blocksByAddress.flatMap(_._2).map(_._2).sum
        (reader, partitionSize)

      case prps: PartialReducerPartitionSpec =>
        val reader = getReader[K, C](
          SparkEnv.get.shuffleManager,
          dependency.shuffleHandle,
          prps.startMapIndex,
          prps.endMapIndex,
          prps.reducerIndex,
          prps.reducerIndex + 1,
          context,
          sqlMetricsReporter)
        val blocksByAddress = shim.getMapSizesByExecutorId(
          dependency.shuffleHandle.shuffleId, 0, Int.MaxValue, prps.reducerIndex,
          prps.reducerIndex + 1)
        val partitionSize = blocksByAddress.flatMap(_._2)
          .filter(tuple => tuple._3 >= prps.startMapIndex && tuple._3 < prps.endMapIndex)
          .map(_._2).sum
        (reader, partitionSize)

      case PartialMapperPartitionSpec(mapIndex, startReducerIndex, endReducerIndex) =>
        val reader = getReader[K, C](
          SparkEnv.get.shuffleManager,
          dependency.shuffleHandle,
          mapIndex,
          mapIndex + 1,
          startReducerIndex,
          endReducerIndex,
          context,
          sqlMetricsReporter)
        val blocksByAddress = shim.getMapSizesByExecutorId(
          dependency.shuffleHandle.shuffleId, 0, Int.MaxValue, startReducerIndex, endReducerIndex)
        val partitionSize = blocksByAddress.flatMap(_._2)
          .filter(_._3 == mapIndex)
          .map(_._2).sum
        (reader, partitionSize)
    }
  }
}
