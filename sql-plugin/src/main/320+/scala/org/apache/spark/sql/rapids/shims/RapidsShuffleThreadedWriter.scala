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

package org.apache.spark.sql.rapids.shims

import org.apache.spark.SparkConf
import org.apache.spark.shuffle.ShuffleWriteMetricsReporter
import org.apache.spark.shuffle.api.{ShuffleExecutorComponents, ShuffleMapOutputWriter}
import org.apache.spark.shuffle.sort.BypassMergeSortShuffleHandle
import org.apache.spark.sql.rapids.RapidsShuffleThreadedWriterBase
import org.apache.spark.storage.{BlockManager, DiskBlockObjectWriter}

class RapidsShuffleThreadedWriter[K, V](
    blockManager: BlockManager,
    handle: BypassMergeSortShuffleHandle[K, V],
    mapId: Long,
    sparkConf: SparkConf,
    writeMetrics: ShuffleWriteMetricsReporter,
    shuffleExecutorComponents: ShuffleExecutorComponents)
  extends RapidsShuffleThreadedWriterBase[K, V](blockManager, handle, mapId, sparkConf,
    writeMetrics, shuffleExecutorComponents)
    with org.apache.spark.shuffle.checksum.ShuffleChecksumSupport {
  // Spark 3.2.0+ computes checksums per map partition as it writes the
  // temporary files to disk. They are stored in a Checksum array.
  private val checksums =
  createPartitionChecksums(handle.dependency.partitioner.numPartitions, sparkConf)

  // Partition lengths, used for MapStatus, but also exposed in Spark 3.2.0+
  private var myPartitionLengths: Array[Long] = null

  override def setChecksumIfNeeded(writer: DiskBlockObjectWriter, partition: Int): Unit = {
    writer.setChecksum(checksums(partition))
  }

  override def getPartitionLengths(): Array[Long] = myPartitionLengths

  override def commitAllPartitions(writer: ShuffleMapOutputWriter): Array[Long] = {
    myPartitionLengths =
      writer.commitAllPartitions(getChecksumValues(checksums))
        .getPartitionLengths
    myPartitionLengths
  }
}

