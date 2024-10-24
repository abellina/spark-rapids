/*
 * Copyright (c) 2020-2024, NVIDIA CORPORATION.
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

import java.io.{File, FileInputStream}
import java.nio.channels.{Channels, FileChannel, WritableByteChannel}
import java.nio.channels.FileChannel.MapMode
import java.nio.file.StandardOpenOption
import java.util.concurrent.ConcurrentHashMap

import ai.rapids.cudf.{Cuda, HostMemoryBuffer, MemoryBuffer}
import com.nvidia.spark.rapids.Arm.{closeOnExcept, withResource}
import com.nvidia.spark.rapids.StorageTier.StorageTier
import com.nvidia.spark.rapids.format.TableMeta
import org.apache.commons.io.IOUtils

import org.apache.spark.TaskContext
import org.apache.spark.sql.rapids.{GpuTaskMetrics, RapidsDiskBlockManager}
import org.apache.spark.sql.rapids.execution.{SerializedHostTableUtils, TrampolineUtil}
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch
import ai.rapids.cudf.DeviceMemoryBuffer

/** A buffer store using files on the local disks. */
class RapidsDiskStore(val diskBlockManager: RapidsDiskBlockManager)
    extends RapidsBufferStoreWithoutSpill(StorageTier.DISK) {
  private[this] val sharedBufferFiles = new ConcurrentHashMap[RapidsBufferId, File]

  private def removeSharedBufferFile(id: RapidsBufferId): Unit = {
    sharedBufferFiles.remove(id)
  }

  private def reportDiskAllocMetrics(metrics: GpuTaskMetrics): String = {
    val taskId = TaskContext.get().taskAttemptId()
    val totalSize = metrics.getDiskBytesAllocated
    val maxSize = metrics.getMaxDiskBytesAllocated
    s"total size for task $taskId is $totalSize, max size is $maxSize"
  }

  private def createBuffer(
      rapidsBuffer: RapidsBuffer,
      channelWriter: ((WritableByteChannel, Cuda.Stream) => (Long, Option[TableMeta])),
      catalog: RapidsBufferCatalog,
      stream: Cuda.Stream,
      base: RapidsMemoryBuffer): RapidsBuffer = {
    // assuming that the disk store gets contiguous buffers
    val id = rapidsBuffer.id
    val path = if (id.canShareDiskPaths) {
      sharedBufferFiles.computeIfAbsent(id, _ => id.getDiskPath(diskBlockManager))
    } else {
      id.getDiskPath(diskBlockManager)
    }

    val (fileOffset, uncompressedSize, diskLength, maybeMeta) = if (id.canShareDiskPaths) {
      // only one writer at a time for now when using shared files
      path.synchronized {
        writeToFile(rapidsBuffer, channelWriter, path, append = true, stream)
      }
    } else {
      writeToFile(rapidsBuffer, channelWriter, path, append = false, stream)
    }
    logDebug(s"Spilled to $path $fileOffset:$diskLength")
    val newDiskBuffer = rapidsBuffer match {
      case rhbb: RapidsHostBatchBuffer =>
        new RapidsDiskColumnarBatch(
          id,
          fileOffset,
          uncompressedSize,
          diskLength,
          rhbb.getSpillPriority,
          catalog,
          base,
          this)
      case rb: RapidsBuffer =>
        maybeMeta.orElse {
          rb match {
            case bufferWithMeta: RapidsBufferWithMeta =>
              Some(bufferWithMeta.meta)
            case _ => None
          }
        } match {
          case Some(tableMeta) =>
            new RapidsDiskBufferWithMeta(
              id,
              fileOffset,
              uncompressedSize,
              diskLength,
              tableMeta,
              rb.getSpillPriority,
              catalog,
              base,
              this)
          case None =>
            new RapidsDiskBuffer(
              id,
              fileOffset,
              uncompressedSize,
              diskLength,
              rb.getSpillPriority,
              catalog,
              base,
              this)
        }
    }
    addBuffer(newDiskBuffer)
    newDiskBuffer
  }

  override def createBuffer(
      rapidsBuffer: RapidsBuffer,
      catalog: RapidsBufferCatalog,
      stream: Cuda.Stream,
      base: RapidsMemoryBuffer): RapidsBuffer = {
    rapidsBuffer match {
      case cw: RapidsBufferChannelWritable =>
        createBuffer(
          rapidsBuffer,
          (channel, stream) => cw.writeToChannel(channel, stream),
          catalog,
          stream,
          base)
      case _ =>
        throw new IllegalStateException(s"${rapidsBuffer} does " +
          s"not support channel writable")
    }
    // TODO: AB
    //TrampolineUtil.incTaskMetricsDiskBytesSpilled(uncompressedSize)
    //val metrics = GpuTaskMetrics.get
    //metrics.incDiskBytesAllocated(uncompressedSize)
    //logDebug(s"acquiring resources for disk buffer $id of size $uncompressedSize bytes")
    //logDebug(reportDiskAllocMetrics(metrics))
    //Some(buff)
  }

  /**
   * Copy a host buffer to a file. It leverages [[RapidsSerializerManager]] from
   * [[RapidsDiskBlockManager]] to do compression or encryption if needed.
   *
   * @param incoming the rapid buffer to be written into a file
   * @param path     file path
   * @param append   whether to append or written into the beginning of the file
   * @param stream   cuda stream
   * @return a tuple of file offset, memory byte size and written size on disk. File offset is where
   *         buffer starts in the targeted file path. Memory byte size is the size of byte buffer
   *         occupied in memory before writing to disk. Written size on disk is actual byte size
   *         written to disk.
   */
  private def writeToFile(rapidsBuffer: RapidsBuffer,
                          channelWriter: ((WritableByteChannel, Cuda.Stream) =>
                             (Long, Option[TableMeta])),
                          path: File,
                          append: Boolean,
                          stream: Cuda.Stream): (Long, Long, Long, Option[TableMeta]) = {
    val option = if (append) {
      Array(StandardOpenOption.CREATE, StandardOpenOption.APPEND)
    } else {
      Array(StandardOpenOption.CREATE, StandardOpenOption.WRITE)
    }
    GpuTaskMetrics.get.spillToDiskTime {
      withResource(FileChannel.open(path.toPath, option: _*)) { fc =>
        val currentPos = fc.position()
        val (writtenBytes, maybeMeta) =
          withResource(Channels.newOutputStream(fc)) { os =>
            withResource(diskBlockManager.getSerializerManager()
              .wrapStream(rapidsBuffer.id, os)) { cos =>
              val outputChannel = Channels.newChannel(cos)
              channelWriter(outputChannel, stream)
            }
          }
        (currentPos, writtenBytes, path.length() - currentPos, maybeMeta)
      }
    }
  }

  /**
   * A RapidsDiskBuffer that is meant to represent just a buffer without meta.
   */
  class RapidsDiskBuffer(id: RapidsBufferId,
                         fileOffset: Long,
                         uncompressedSize: Long,
                         onDiskSizeInBytes: Long,
                         spillPriority: Long,
                         catalog: RapidsBufferCatalog,
                         override val base: RapidsMemoryBuffer,
                         diskStore: RapidsDiskStore)
    extends RapidsBufferBase(id, spillPriority) {

    def getCopyIterator(stream: Cuda.Stream): RapidsBufferCopyIterator =
      new RapidsBufferCopyIterator(
        singleShotBuffer = Some(getMemoryBuffer(stream)))

    override def copyTo(store: RapidsBufferStore, stream: Cuda.Stream): RapidsBuffer = {
      store.createBuffer(this, catalog, stream, base)
    }

    // FIXME: Need to be clean up. Tracked in https://github.com/NVIDIA/spark-rapids/issues/9496
    override val memoryUsedBytes: Long = uncompressedSize

    override def getDeviceMemoryBuffer(stream: Cuda.Stream): DeviceMemoryBuffer = {
      withResource(getMemoryBuffer(stream)) { hb =>
        closeOnExcept(DeviceMemoryBuffer.allocate(hb.getLength, stream)) { db =>
          db.copyFromMemoryBuffer(0, hb, 0, db.getLength, stream)
          db
        }
      }
    }

    // used in UCX shuffle and copy iterator
    override def getMemoryBuffer(stream: Cuda.Stream): MemoryBuffer = synchronized {
      require(onDiskSizeInBytes > 0,
        s"$this attempted an invalid 0-byte mmap of a file")
      val diskBlockManager = diskStore.diskBlockManager
      val path = id.getDiskPath(diskBlockManager)
      val serializerManager = diskBlockManager.getSerializerManager()
      val memBuffer = if (serializerManager.isRapidsSpill(id)) {
        // Only go through serializerManager's stream wrapper for spill case
        closeOnExcept(HostAlloc.alloc(uncompressedSize)) {
          decompressed => GpuTaskMetrics.get.readSpillFromDiskTime {
            withResource(FileChannel.open(path.toPath, StandardOpenOption.READ)) { c =>
              c.position(fileOffset)
              withResource(Channels.newInputStream(c)) { compressed =>
                withResource(serializerManager.wrapStream(id, compressed)) { in =>
                  withResource(new HostMemoryOutputStream(decompressed)) { out =>
                    IOUtils.copy(in, out)
                  }
                  decompressed
                }
              }
            }
          }
        }
      } else {
        // Reserved mmap read fashion for UCX shuffle path. Also it's skipping encryption and
        // compression.
        HostMemoryBuffer.mapFile(path, MapMode.READ_WRITE, fileOffset, onDiskSizeInBytes)
      }
      memBuffer
    }

    override def getHostMemoryBuffer(stream: Cuda.Stream): HostMemoryBuffer =
      getMemoryBuffer(stream).asInstanceOf[HostMemoryBuffer]

    override def close(): Unit = synchronized {
      super.close()
    }

    override protected def releaseResources(): Unit = {
      // Buffers that share paths must be cleaned up elsewhere
      if (id.canShareDiskPaths) {
        diskStore.removeSharedBufferFile(id)
      } else {
        val path = id.getDiskPath(diskStore.diskBlockManager)
        if (!path.delete() && path.exists()) {
          logWarning(s"Unable to delete spill path $path")
        }
      }
    }
  }

  /**
   * A RapidsDiskBuffer that is mean to represent device-bound memory. This
   * buffer can produce a device-backed ColumnarBatch.
   */
  class RapidsDiskBufferWithMeta(id: RapidsBufferId,
                                 fileOffset: Long,
                                 uncompressedSize: Long,
                                 onDiskSizeInBytes: Long,
                                 _meta: TableMeta,
                                 spillPriority: Long,
                                 catalog: RapidsBufferCatalog,
                                 override val base: RapidsMemoryBuffer,
                                 diskStore: RapidsDiskStore)
    extends RapidsDiskBuffer(id, fileOffset,
        uncompressedSize, onDiskSizeInBytes, spillPriority, catalog, base, diskStore)
      with RapidsBufferWithMeta
      with CopyableRapidsBuffer {

    override def meta: TableMeta = _meta
  }

  /**
   * A RapidsDiskBuffer that should remain in the host, producing host-backed
   * ColumnarBatch if the caller invokes getHostColumnarBatch, but not producing
   * anything on the device.
   */
  class RapidsDiskColumnarBatch(id: RapidsBufferId,
                                fileOffset: Long,
                                size: Long,
                                uncompressedSize: Long,
                                spillPriority: Long,
                                catalog: RapidsBufferCatalog,
                                override val base: RapidsMemoryBuffer,
                                diskStore: RapidsDiskStore)
    extends RapidsDiskBuffer(
      id, fileOffset, size, uncompressedSize, spillPriority, catalog, base, diskStore)
      with RapidsHostBatchBuffer {

    override def getHostColumnarBatch(sparkTypes: Array[DataType]): ColumnarBatch = {
      require(fileOffset == 0,
        "Attempted to obtain a HostColumnarBatch from a spilled RapidsBuffer that is sharing " +
          "paths on disk")
      val path = id.getDiskPath(diskStore.diskBlockManager)
      withResource(new FileInputStream(path)) { fis =>
        withResource(diskStore.diskBlockManager.getSerializerManager()
          .wrapStream(id, fis)) { fs =>
          val (header, hostBuffer) = SerializedHostTableUtils.readTableHeaderAndBuffer(fs)
          val hostCols = withResource(hostBuffer) { _ =>
            SerializedHostTableUtils.buildHostColumns(header, hostBuffer, sparkTypes)
          }
          new ColumnarBatch(hostCols.toArray, header.getNumRows)
        }
      }
    }
  }
}

