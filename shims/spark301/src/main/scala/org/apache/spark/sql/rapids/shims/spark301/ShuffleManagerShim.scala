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
package org.apache.spark.sql.rapids.shims.spark301

import com.nvidia.spark.rapids.ShimLoader

import org.apache.spark.TaskContext
import org.apache.spark.shuffle.{ShuffleHandle, ShuffleManager, ShuffleReader, ShuffleReadMetricsReporter}
import org.apache.spark.sql.execution.{CoalescedPartitionSpec, PartialMapperPartitionSpec, PartialReducerPartitionSpec}
import org.apache.spark.sql.rapids.execution.ShuffledBatchRDDPartition
import org.apache.spark.sql.rapids.{GpuPartialReducerPartitionSpec, ShuffleManagerShimBase}

class ShuffleManagerShim extends ShuffleManagerShimBase {
  override def getReaderAndPartitionSize[K, C](
      shuffleManager: ShuffleManager,
      shuffleHandle: ShuffleHandle,
      taskContext: TaskContext,
      metrics: ShuffleReadMetricsReporter,
      shuffledBatchRDDPartition: ShuffledBatchRDDPartition): (ShuffleReader[K, C], Long) = {

    @transient val shim = ShimLoader.getSparkShims
    shuffledBatchRDDPartition.spec match {
      case CoalescedPartitionSpec(startReducerIndex, endReducerIndex) =>
        val reader = shuffleManager.getReader[K,C](
          shuffleHandle,
          startReducerIndex,
          endReducerIndex,
          taskContext,
          metrics)
        val blocksByAddress = shim.getMapSizesByExecutorId(
          shuffleHandle.shuffleId, 0, Int.MaxValue, startReducerIndex, endReducerIndex)
        val partitionSize = blocksByAddress.flatMap(_._2).map(_._2).sum
        (reader, partitionSize)

      case PartialReducerPartitionSpec(reducerIndex, startMapIndex, endMapIndex) =>
        val reader = shuffleManager.getReaderForRange[K,C](
          shuffleHandle,
          startMapIndex,
          endMapIndex,
          reducerIndex,
          reducerIndex + 1,
          taskContext,
          metrics)

        val blocksByAddress = shim.getMapSizesByExecutorId(
          shuffleHandle.shuffleId, 0, Int.MaxValue, reducerIndex, reducerIndex + 1)
        val partitionSize = blocksByAddress.flatMap(_._2)
            .filter(tuple => tuple._3 >= startMapIndex && tuple._3 < endMapIndex)
            .map(_._2).sum
        (reader, partitionSize)

      case PartialMapperPartitionSpec(mapIndex, startReducerIndex, endReducerIndex) =>
        val reader = shuffleManager.getReaderForRange[K,C](
          shuffleHandle,
          mapIndex,
          mapIndex + 1,
          startReducerIndex,
          endReducerIndex,
          taskContext,
          metrics)
        val blocksByAddress = shim.getMapSizesByExecutorId(
          shuffleHandle.shuffleId, 0, Int.MaxValue, startReducerIndex, endReducerIndex)
        val partitionSize = blocksByAddress.flatMap(_._2)
            .filter(_._3 == mapIndex)
            .map(_._2).sum
        (reader, partitionSize)
    }
  }

  override def toGpu(x: PartialReducerPartitionSpec): GpuPartialReducerPartitionSpec = {
    GpuPartialReducerPartitionSpec(x.reducerIndex, x.startMapIndex, x.endMapIndex, -1L)
  }
}
