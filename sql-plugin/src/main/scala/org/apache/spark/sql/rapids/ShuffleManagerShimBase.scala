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

package org.apache.spark.sql.rapids

import org.apache.spark.TaskContext
import org.apache.spark.shuffle.{ShuffleHandle, ShuffleManager, ShuffleReader, ShuffleReadMetricsReporter}
import org.apache.spark.sql.execution.PartialReducerPartitionSpec
import org.apache.spark.sql.rapids.execution.ShuffledBatchRDDPartition

case class GpuPartialReducerPartitionSpec(
  reducerIndex: Int, startMapIndex: Int, endMapIndex: Int,
  @transient dataSize: Long
)

trait ShuffleManagerShimBase {

  def getReaderAndPartitionSize[K, C](
      shuffleManager: ShuffleManager,
      shuffleHandle: ShuffleHandle,
      taskContext: TaskContext,
      metrics: ShuffleReadMetricsReporter,
      shuffledBatchRDDPartition: ShuffledBatchRDDPartition): (ShuffleReader[K, C], Long)

  // conversions from Spark to Gpu classes
  def toGpu(x: PartialReducerPartitionSpec): GpuPartialReducerPartitionSpec
}
