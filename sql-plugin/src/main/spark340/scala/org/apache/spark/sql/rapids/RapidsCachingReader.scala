/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

/*** spark-rapids-shim-json-lines
{"spark": "340"}
{"spark": "341"}
{"spark": "341db"}
{"spark": "342"}
{"spark": "350"}
{"spark": "351"}
spark-rapids-shim-json-lines ***/
package org.apache.spark.sql.rapids

import scala.collection
import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.{NvtxColor, NvtxRange}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.shuffle.{RapidsShuffleIterator, RapidsShuffleTransport}

import org.apache.spark.{InterruptibleIterator, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.shuffle.{ShuffleReader, ShuffleReadMetricsReporter}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage.{BlockId, BlockManagerId, ShuffleBlockBatchId, ShuffleBlockId}
import org.apache.spark.util.CompletionIterator

trait ShuffleMetricsUpdater {
  /**
   * Trait used as a way to expose the `ShuffleReadMetricsReporter` to the iterator.
   * @param fetchWaitTimeInMs this matches the CPU name (except for the units) but it is actually
   *                          the aggreagate amount of time a task is blocked, not working on
   *                          anything, waiting for data.
   * @param remoteBlocksFetched aggregate of number of `ShuffleBlockId`s fetched.
   * @param remoteBytesRead aggregate size of all contiguous buffers received
   * @param rowsFetched aggregate of number of rows received
   */
  def update(
    fetchWaitTimeInMs: Long,
    remoteBlocksFetched: Long,
    remoteBytesRead: Long,
    rowsFetched: Long): Unit
}

class RapidsBufferCoalesceIterator(
    catalog: ShuffleBufferCatalog,
    receiveCatalog: ShuffleReceivedBufferCatalog,
    rapidsBufferCacheIt: Iterator[RapidsBufferHandle],
    rapidsBufferUcxIt: Iterator[RapidsBufferHandle],
    sparkTypes: Array[DataType],
    metrics: ShuffleReadMetricsReporter) extends Iterator[ColumnarBatch] {
  override def hasNext(): Boolean = rapidsBufferCacheIt.hasNext || rapidsBufferUcxIt.hasNext

  override def next(): ColumnarBatch = {
    val toConcat = new ArrayBuffer[(ByteBuffer, DeviceMemoryBuffer)]()
    val acquired = new ArrayBuffer[RapidsBuffer]()
    val toRemove = new ArrayBuffer[RapidsBuffer]()
    var numBytes = 0L
    withResource(new NvtxRange("new concat", NvtxColor.DARK_GREEN)) { _ =>
      while (rapidsBufferCacheIt.hasNext && numBytes < 2L * 1024 * 1024 * 1024) {
        val handle = rapidsBufferCacheIt.next()
        val buffer = catalog.acquireBuffer(handle)
        numBytes += buffer.memoryUsedBytes
        toConcat.append(buffer.getMetaAndBuffer)
        acquired.append(buffer)
      }
      while (rapidsBufferUcxIt.hasNext && numBytes < 2L * 1024 * 1024 * 1024) {
        val handle = rapidsBufferUcxIt.next()
        val buffer = receiveCatalog.acquireBuffer(handle)
        numBytes += buffer.memoryUsedBytes
        toConcat.append(buffer.getMetaAndBuffer)
        toRemove.append(buffer)
        acquired.append(handle)
      }
      val concatenated = ai.rapids.cudf.Table.concatenatePacked(
        toConcat.map(_._1).toArray,
        toConcat.map(_._2).toArray)
      acquired.foreach(_.close())
      toRemove.foreach(h => receiveCatalog.removeBuffer(h))
      val res = GpuColumnVector.from(concatenated, sparkTypes)
      metrics.incLocalBytesRead(numBytes)
      metrics.incRecordsRead(res.numRows())
      res
    }
  }
}

class RapidsCachingReader[K, C](
    rapidsConf: RapidsConf,
    localId: BlockManagerId,
    blocksByAddress: Iterator[(BlockManagerId, collection.Seq[(BlockId, Long, Int)])],
    context: TaskContext,
    metrics: ShuffleReadMetricsReporter,
    transport: Option[RapidsShuffleTransport],
    catalog: ShuffleBufferCatalog,
    sparkTypes: Array[DataType])
  extends ShuffleReader[K, C] with Logging {

  override def read(): Iterator[Product2[K, C]] = {
    val readRange = new NvtxRange(s"RapidsCachingReader.read", NvtxColor.DARK_GREEN)
    try {
      val blocksForRapidsTransport =
          new ArrayBuffer[(BlockManagerId, collection.Seq[(BlockId, Long, Int)])]()
      val cachedBlocks = new ArrayBuffer[BlockId]()
      val cachedBufferHandles = new ArrayBuffer[RapidsBufferHandle]()
      val blocksByAddressMap: Map[BlockManagerId, collection.Seq[(BlockId, Long, Int)]] = 
          blocksByAddress.toMap

      blocksByAddressMap.keys.foreach(blockManagerId => {
        val blockInfos: collection.Seq[(BlockId, Long, Int)] = blocksByAddressMap(blockManagerId)

        logDebug("Trying to read block from manager: " + blockManagerId)
        if (blockManagerId.executorId == localId.executorId) {
          val readLocalRange = new NvtxRange("Read Local", NvtxColor.GREEN)
          try {
            blockInfos.foreach(
              blockInfo => {
                val blockId = blockInfo._1
                val shuffleBufferHandles: IndexedSeq[RapidsBufferHandle] = blockId match {
                  case sbbid: ShuffleBlockBatchId =>
                    (sbbid.startReduceId to sbbid.endReduceId).flatMap { reduceId =>
                      cachedBlocks.append(blockId)
                      val sBlockId = ShuffleBlockId(sbbid.shuffleId, sbbid.mapId, reduceId)
                      catalog.blockIdToBufferHandles(sBlockId)
                    }
                  case sbid: ShuffleBlockId =>
                    cachedBlocks.append(blockId)
                    catalog.blockIdToBufferHandles(sbid)
                  case _ => throw new IllegalArgumentException(
                    s"${blockId.getClass} $blockId is not currently supported")
                }

                cachedBufferHandles ++= shuffleBufferHandles

                // Update the spill priorities of these buffers to indicate they are about
                // to be read and therefore should not be spilled if possible.
                shuffleBufferHandles.foreach(catalog.updateSpillPriorityForLocalRead)

                if (shuffleBufferHandles.nonEmpty) {
                  metrics.incLocalBlocksFetched(1)
                }
              })
          } finally {
            readLocalRange.close()
          }
        } else {
          require(
            blockManagerId.topologyInfo.isDefined &&
              blockManagerId.topologyInfo.get
                .startsWith(s"${RapidsShuffleTransport.BLOCK_MANAGER_ID_TOPO_PREFIX}="), {
              val enabledHint = if (rapidsConf.isUCXShuffleManagerMode) {
                "The shuffle transport is disabled. " +
                    s"Please set ${RapidsConf.SHUFFLE_MANAGER_MODE.key}=UCX to enable " +
                    "fetching remote blocks."
              } else {
                "This is unexpected behavior!"
              }
              s"Attempting to handle non-rapids enabled blocks from $blockManagerId. ${enabledHint}"
            })
          blocksForRapidsTransport.append((blockManagerId, blockInfos))
        }
      })

      logInfo(s"Will read ${cachedBlocks.size} cached blocks, " +
        s"${blocksForRapidsTransport.size} remote blocks from the RapidsShuffleTransport. ")

      if (transport.isEmpty && blocksForRapidsTransport.nonEmpty) {
        throw new IllegalStateException("Had blocks marked for use with the " +
          "RapidsShuffleTransport, but the transport was not initialized")
      }

      val metricsUpdater = new ShuffleMetricsUpdater {
        override def update (fetchWaitTimeInMs: Long, remoteBlocksFetched: Long,
            remoteBytesRead: Long, rowsFetched: Long): Unit = {
          metrics.incFetchWaitTime(fetchWaitTimeInMs)
          metrics.incRemoteBlocksFetched(remoteBlocksFetched)
          metrics.incRemoteBytesRead(remoteBytesRead)
          metrics.incRecordsRead(rowsFetched)
        }
      }

      val itRange = new NvtxRange("Shuffle Iterator prep", NvtxColor.BLUE)
      try {
        val cbArrayFromUcx = if (blocksForRapidsTransport.nonEmpty) {
          val rapidsShuffleIterator = new RapidsShuffleIterator(localId, rapidsConf, transport.get,
            blocksForRapidsTransport.toArray, metricsUpdater, sparkTypes, context.taskAttemptId())
          rapidsShuffleIterator.start()
          rapidsShuffleIterator
        } else {
          Iterator.empty
        }

        val combinedIt = new RapidsBufferCoalesceIterator(
          catalog,
          GpuShuffleEnv.getReceivedCatalog,
          cachedBufferHandles.iterator,
          rapidsShuffleIterator,
          sparkTypes,
          metrics)

        val completionIter = CompletionIterator[(K, C), Iterator[(K, C)]](
          combinedIt.map{cb => (0, cb)}.asInstanceOf[Iterator[(K, C)]], {
            context.taskMetrics().mergeShuffleReadMetrics()
          })

        new InterruptibleIterator[(K, C)](context, completionIter)

      } finally {
        itRange.close()
      }
    } finally {
      readRange.close()
    }
  }
}
