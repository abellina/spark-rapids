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

package org.apache.spark.sql.rapids

import scala.collection.mutable.ArrayBuffer
import ai.rapids.cudf.{NvtxColor, NvtxRange}
import com.nvidia.spark.rapids._
import com.nvidia.spark.rapids.shuffle.{RapidsShuffleIterator, RapidsShuffleTransport}
import org.apache.spark.{InterruptibleIterator, TaskContext}
import org.apache.spark.internal.Logging
import org.apache.spark.shuffle.{ShuffleReadMetricsReporter, ShuffleReader}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage.{BlockId, BlockManagerId, ShuffleBlockBatchId, ShuffleBlockId}
import org.apache.spark.util.CompletionIterator

trait ColumnarBatchProvider extends AutoCloseable {
  def getBatch: ColumnarBatch
  def numRows: Int
  def sizeInBytes: Long
  override def close(): Unit = {}
}

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
    rowsFetched: Long)
}

class RapidsCachingReader[K, C](
    rapidsConf: RapidsConf,
    localId: BlockManagerId,
    blocksByAddress: Iterator[(BlockManagerId, Seq[(BlockId, Long, Int)])],
    context: TaskContext,
    metrics: ShuffleReadMetricsReporter,
    transport: Option[RapidsShuffleTransport],
    catalog: ShuffleBufferCatalog,
    sparkTypes: Array[DataType])
  extends ShuffleReader[K, C]  with Arm with Logging {

  override def read(): Iterator[Product2[K, C]] = {
    val readRange = new NvtxRange(s"RapidsCachingReader.read", NvtxColor.DARK_GREEN)
    try {
      val blocksForRapidsTransport = new ArrayBuffer[(BlockManagerId, Seq[(BlockId, Long, Int)])]()
      val cachedBlocks = new ArrayBuffer[BlockId]()
      val cachedBufferIds = new ArrayBuffer[ShuffleBufferId]()
      val blocksByAddressMap: Map[BlockManagerId, Seq[(BlockId, Long, Int)]] = blocksByAddress.toMap

      blocksByAddressMap.keys.foreach(blockManagerId => {
        val blockInfos: Seq[(BlockId, Long, Int)] = blocksByAddressMap(blockManagerId)

        logDebug("Trying to read block from manager: " + blockManagerId)
        if (blockManagerId.executorId == localId.executorId) {
          val readLocalRange = new NvtxRange("Read Local", NvtxColor.GREEN)
          try {
            blockInfos.foreach(
              blockInfo => {
                val blockId = blockInfo._1
                val shuffleBufferIds: IndexedSeq[ShuffleBufferId] = blockId match {
                  case sbbid: ShuffleBlockBatchId =>
                    (sbbid.startReduceId to sbbid.endReduceId).flatMap { reduceId =>
                      cachedBlocks.append(blockId)
                      val sBlockId = ShuffleBlockId(sbbid.shuffleId, sbbid.mapId, reduceId)
                      catalog.blockIdToBuffersIds(sBlockId)
                    }
                  case sbid: ShuffleBlockId =>
                    cachedBlocks.append(blockId)
                    catalog.blockIdToBuffersIds(sbid)
                  case _ => throw new IllegalArgumentException(
                    s"${blockId.getClass} $blockId is not currently supported")
                }

                cachedBufferIds ++= shuffleBufferIds

                // Update the spill priorities of these buffers to indicate they are about
                // to be read and therefore should not be spilled if possible.
                shuffleBufferIds.foreach(catalog.updateSpillPriorityForLocalRead)

                if (shuffleBufferIds.nonEmpty) {
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
        val cachedIt = cachedBufferIds.iterator.map(bufferId => {
          // No good way to get a metric in here for semaphore wait time
          GpuSemaphore.acquireIfNecessary(context, NoopMetric)
          // TODO: make the cached batch highest priority to spill
          new ColumnarBatchProvider {
            override def sizeInBytes: Long = {
              withResource(catalog.acquireBuffer(bufferId)) { buffer =>
                buffer.size
              }
            }
            override def numRows: Int = {
              withResource(catalog.acquireBuffer(bufferId)) { buffer =>
                buffer.withColumnarBatch(sparkTypes) { cb =>
                  cb.numRows()
                }
              }
            }

            override def getBatch: ColumnarBatch = {
              withResource(catalog.acquireBuffer(bufferId)) { buffer =>
                buffer.withColumnarBatch(sparkTypes) { cb =>
                  val cachedBytesRead = GpuColumnVector.getTotalDeviceMemoryUsed(cb)
                  metrics.incLocalBytesRead(cachedBytesRead)
                  metrics.incRecordsRead(cb.numRows())
                }
                buffer.getColumnarBatch(sparkTypes)
              }
            }
          }
        })

        val cbArrayFromUcx = if (blocksForRapidsTransport.nonEmpty) {
          new RapidsShuffleIterator(localId, rapidsConf, transport.get,
            blocksForRapidsTransport.toArray, metricsUpdater, sparkTypes)
        } else {
          Iterator.empty
        }

        val rapidsBufferIdIter = cachedIt ++ cbArrayFromUcx


        val materializeAndConcat = new AbstractSpillableCoalesceIterator(
          rapidsBufferIdIter,
          TargetSize(rapidsConf.gpuTargetBatchSizeBytes),
          NoopMetric,
          NoopMetric,
          NoopMetric,
          NoopMetric,
          NoopMetric,
          NoopMetric,
          NoopMetric,
          "ucx_coalesce") {

          val batches: ArrayBuffer[ColumnarBatchProvider] = ArrayBuffer.empty
          var onDeck: Option[ColumnarBatchProvider] = None

          override def addBatchToConcat(batch: ColumnarBatchProvider): Unit = {
            batches.append(batch)
          }

          override def cleanupConcatIsDone(): Unit = {
            // TODO: set peak dev memory metric
            batches.clear()
          }

          override protected def clearOnDeck(): Unit = {
            onDeck.foreach(_.close)
            onDeck = None
          }

          override def concatAllAndPutOnGPU(): ColumnarBatch = {
            val batchesToConcat = batches.toArray.map(_.getBatch)
            batches.clear()
            ConcatAndConsumeAll.buildNonEmptyBatchFromTypes(
                batchesToConcat, sparkTypes)
          }

          override protected def hasOnDeck: Boolean = {
            onDeck.isDefined
          }

          override protected def saveOnDeck(batch: ColumnarBatchProvider): Unit = {
            onDeck = Some(batch)
          }

          override protected def popOnDeck(): ColumnarBatchProvider  = {
            val res = onDeck.get
            clearOnDeck()
            res
          }

          override def initNewBatch(batch: ColumnarBatchProvider): Unit = {
            batches.foreach(_.close()) // TODO: use safeClose
            batches.clear()
          }
        }

        val completionIter = CompletionIterator[(K, C), Iterator[(K, C)]](
          materializeAndConcat.map { cb => (0, cb) }.asInstanceOf[Iterator[(K, C)]], {
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
