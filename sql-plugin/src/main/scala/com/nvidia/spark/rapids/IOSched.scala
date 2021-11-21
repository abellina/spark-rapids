package com.nvidia.spark.rapids

import ai.rapids.cudf.{ContiguousTable, HostMemoryBuffer, Table}
import org.apache.hadoop.fs.Path
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.types.StructType

import java.util.concurrent.{ConcurrentHashMap, Executors, LinkedBlockingQueue}
import java.util.concurrent.atomic.AtomicLong
import scala.collection.mutable.ArrayBuffer

object IOSched extends Logging with Arm {
  type WriteHeaderFn = (HostMemoryBuffer => Long)
  type StreamerFn = (HostMemoryBuffer, Long) => (Long, Long, ArrayBuffer[DataBlockBase])
  type CombinerFn =
    (HostMemoryBuffer, ArrayBuffer[DataBlockBase], SchemaBase, Long, Long, Long)
    => (HostMemoryBuffer, Long)

  type ReadToTableFn = (HostMemoryBuffer, Long, SchemaBase, ExtraInfo) =>
    Table

  case class Task(myIx: Long,
                  initTotalSize: Long,
                  writeHeaderFn: WriteHeaderFn,
                  stramerFn: StreamerFn,
                  combiner: CombinerFn,
                  readToTableFn: ReadToTableFn,
                  clippedSchema: SchemaBase,
                  extraInfo: ExtraInfo)

  val q = new LinkedBlockingQueue[Task]()
  val r = new ConcurrentHashMap[Long, ContiguousTable]()
  var ix = new AtomicLong(0L)
  val bMonitor = new Object

  def getMySlice(l: Long): ContiguousTable = {
    while (!r.containsKey(l)) {
      bMonitor.synchronized {
        logInfo(s"Waiting for ${l}")
        bMonitor.wait()
      }
    }
    r.remove(l)
  }

  val tp = Executors.newSingleThreadExecutor()

  tp.execute(() => {
    while (true) {
      val toCompute = new ArrayBuffer[Task]()
      var initialTotalSize = 0L
      while (q.size() > 0 && initialTotalSize < 1L*1024*1024*1024) {
        val task = q.take()
        initialTotalSize += task.initTotalSize
        toCompute.append(task)
      }
      if (toCompute.nonEmpty) {
        val hmb = HostMemoryBuffer.allocate(initialTotalSize)
        var offset = toCompute.head.writeHeaderFn(hmb)
        val allOutBlocks = new ArrayBuffer[DataBlockBase]()
        var wholeBufferSize = 0L

        val rowCountsPerTask = new ArrayBuffer[Int]()
        var runningSumOfRows = 0
        for (res <- toCompute) {
          val (bufferSize, newOffset, outBlocks) = res.stramerFn(hmb, offset)
          offset = newOffset
          allOutBlocks ++= outBlocks
          wholeBufferSize += bufferSize
          val rowCount = outBlocks.map(_.getRowCount).sum.toInt
          logInfo(s"Working! on ${res} blocks = ${outBlocks.size} rowCount ${rowCount}")
          runningSumOfRows += rowCount
          rowCountsPerTask.append(runningSumOfRows)
        }
        rowCountsPerTask.remove(rowCountsPerTask.size-1)
        logInfo(s"Built bigger file with whole size = ${wholeBufferSize}," +
          s" init size ${initialTotalSize}, splits ${rowCountsPerTask}, " +
          s"blocks ${allOutBlocks.size}, rowCount ${runningSumOfRows}")
        val (dataBuffer, dataSize) = toCompute.head.combiner(
          hmb, allOutBlocks, toCompute.head.clippedSchema,
          wholeBufferSize, initialTotalSize, offset)

        withResource(dataBuffer) { _ =>
          val tbl = toCompute.head.readToTableFn(dataBuffer, dataSize,
            toCompute.head.clippedSchema, toCompute.head.extraInfo)
          withResource(tbl) { _ =>
            val tbls: Array[ContiguousTable] = tbl.contiguousSplit(rowCountsPerTask: _*)
            logInfo(s"Got slices ${tbls.length} have tasks waiting ${toCompute.length}. Counts " +
              s"${rowCountsPerTask}")
            var i = 0
            for (res <- toCompute) {
              r.put(res.myIx, tbls(i))
              i += 1
            }
          }
        }
      }
      bMonitor.synchronized {
        bMonitor.notifyAll()
      }
    }
  })

  def addReadTask(clippedSchema: SchemaBase,
                  initTotalSize: Long,
                  extraInfo: ExtraInfo,
                  writeHeaderFn: WriteHeaderFn,
                  blockStreamerFn: StreamerFn,
                  blockCombinerFn: CombinerFn,
                  readerToTableFn: ReadToTableFn,
                  readDataSchema: StructType,
                  currentChunkedBlocks: Seq[(Path, DataBlockBase)]): Option[ContiguousTable] = {
    logInfo(s"I got a task from ${TaskContext.get()}: " +
      s"initTotalSize ${initTotalSize} " +
      s"with schema ${clippedSchema} and extraInfo ${extraInfo}")

    val myIx: Long = ix.getAndIncrement()
    q.offer(Task(
      myIx,
      initTotalSize,
      writeHeaderFn,
      blockStreamerFn,
      blockCombinerFn,
      readerToTableFn,
      clippedSchema,
      extraInfo))

    Some(getMySlice(myIx))
  }
}
