/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

package org.apache.spark.sql.rapids.shims.spark330

import ai.rapids.cudf.{NvtxColor, NvtxRange}

import org.apache.spark.{SparkConf, SparkEnv}
import org.apache.spark.shuffle._
import org.apache.spark.sql.rapids.{ProxyRapidsShuffleInternalManagerBase, RapidsShuffleInternalManagerBase}
import org.apache.spark.storage.{BlockId, BlockManagerId}

/**
 * A shuffle manager optimized for the RAPIDS Plugin For Apache Spark.
 * @note This is an internal class to obtain access to the private
 *       `ShuffleManager` and `SortShuffleManager` classes.
 */
class RapidsShuffleInternalManager(conf: SparkConf, isDriver: Boolean)
    extends RapidsShuffleInternalManagerBase(conf, isDriver) {
  override def getMapSizes[K, C](
      handle: BaseShuffleHandle[K, C, C],
      startMapIndex: Int,
      endMapIndex: Int,
      startPartition: Int,
      endPartition: Int): (Iterator[(BlockManagerId, Seq[(BlockId, Long, Int)])], Boolean) = {
    val shuffleId = handle.shuffleId
    withResource(new NvtxRange("getMapSizesByExecId", NvtxColor.CYAN)) { _ =>
      if (handle.dependency.isShuffleMergeFinalizedMarked) {
        val res = SparkEnv.get.mapOutputTracker.getPushBasedShuffleMapSizesByExecutorId(
          handle.shuffleId, startMapIndex, endMapIndex, startPartition, endPartition)
        (res.iter, res.enableBatchFetch)
      } else {
        val address = SparkEnv.get.mapOutputTracker.getMapSizesByExecutorId(
          handle.shuffleId, startMapIndex, endMapIndex, startPartition, endPartition)
        (address, true)
      }
    }
  }
}

class ProxyRapidsShuffleInternalManager(conf: SparkConf, isDriver: Boolean)
  extends ProxyRapidsShuffleInternalManagerBase(conf, isDriver)
    with ShuffleManager
