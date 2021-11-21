package com.nvidia.spark.rapids

import ai.rapids.cudf.{ContiguousTable, HostMemoryBuffer, Table}
import org.apache.hadoop.fs.Path
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.types.StructType

import java.util.concurrent.{ConcurrentHashMap, Executors, LinkedBlockingQueue}
import java.util.concurrent.atomic.AtomicLong
import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import java.util.concurrent.Future

object IOSched extends Logging with Arm {
  type StreamedResult = (HostMemoryBuffer, Long, Long, Seq[DataBlockBase])
  type WriteHeaderFn = (HostMemoryBuffer => Long)
  type StreamerFn = (HostMemoryBuffer, Long) => (Long, ArrayBuffer[DataBlockBase])
  type CombinerFn =
    (HostMemoryBuffer, Seq[DataBlockBase], SchemaBase, Long, Long, Long)
    => (HostMemoryBuffer, Long)

  type ReadToTableFn = (HostMemoryBuffer, Long, SchemaBase, ExtraInfo) =>
    Table

  case class Task(filesAndBlocks: mutable.LinkedHashMap[Path, ArrayBuffer[DataBlockBase]],
                  myIx: Long,
                  initTotalSize: Long,
                  writeHeaderFn: WriteHeaderFn,
                  streamerFn: StreamerFn,
                  combiner: CombinerFn,
                  readToTableFn: ReadToTableFn,
                  finalFn: ((Long, Seq[DataBlockBase], SchemaBase) => Long),
                  clippedSchema: SchemaBase,
                  extraInfo: ExtraInfo)

  val q = new LinkedBlockingQueue[Task]()
  val b = new LinkedBlockingQueue[() => Unit]()
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
  val tp2 = Executors.newFixedThreadPool(20)

  tp.execute(() => {
    while (true) {
      val toCompute = new ArrayBuffer[Task]()
      var initialTotalSize = 0L
      var schema: SchemaBase = null
      var schemasSame: Boolean = true
      while (q.size() > 0 &&
        initialTotalSize < 1L*1024*1024*1024 &&
        schemasSame) {
        logInfo("Dequeing...")
        if (schema != null) {
          logInfo("Dequeing: schema not null")
          schemasSame = q.peek().clippedSchema == schema
          logInfo(s"Dequeing: schema not null:: same? ${schemasSame}")
        }
        if (schemasSame) {
          logInfo(s"Take it!")
          val task = q.take()
          initialTotalSize += task.initTotalSize
          toCompute.append(task)
          schema = task.clippedSchema
          logInfo(s"More? ${q.size()}")
        }
      }
      if (toCompute.nonEmpty) {
        val hmb = HostMemoryBuffer.allocate(initialTotalSize)
        val headerOffset = {
          toCompute.head.writeHeaderFn(hmb)
        }
        var offset = headerOffset
        //val allOutBlocks = new ArrayBuffer[DataBlockBase]()
        val rowCountsPerTask = new ArrayBuffer[Int]()
        var runningSumOfRows = 0
        val offsets: Seq[Long] = toCompute.map { res =>
          res.filesAndBlocks.values.toSeq.flatten.map(_.getBlockSize).sum
        }
        val futures = new ArrayBuffer[Future[Seq[DataBlockBase]]]()
        for ((res, myOffset) <- toCompute.zip(offsets)) {
          val theOffset = offset
          futures += tp2.submit(() => {
            val (_, outBlocks: Seq[DataBlockBase]) = res.streamerFn(hmb, theOffset)
            outBlocks
          })
          offset += myOffset
          val rowCount = res.filesAndBlocks.values.flatten.map(_.getRowCount).sum.toInt
          logInfo(s"Working! on ${res} " +
            s"offset so far = ${offset} " +
            s"rowCount = ${rowCount}")
          runningSumOfRows += rowCount
          rowCountsPerTask.append(runningSumOfRows)
        }

        val allOutBlocks: Seq[DataBlockBase] =
          futures.flatMap(f => f.get())

        // Fourth, calculate the final buffer size
        val wholeBufferSize = toCompute.head.finalFn(offset, allOutBlocks, schema)

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

  def addReadTask(filesAndBlocks: mutable.LinkedHashMap[Path, ArrayBuffer[DataBlockBase]],
                  clippedSchema: SchemaBase,
                  initTotalSize: Long,
                  extraInfo: ExtraInfo,
                  writeHeaderFn: WriteHeaderFn,
                  blockStreamerFn: StreamerFn,
                  blockCombinerFn: CombinerFn,
                  readerToTableFn: ReadToTableFn,
                  finalFn: ((Long, Seq[DataBlockBase], SchemaBase) => Long),
                  readDataSchema: StructType,
                  currentChunkedBlocks: Seq[(Path, DataBlockBase)]): Option[ContiguousTable] = {
    logInfo(s"I got a task from ${TaskContext.get()}: " +
      s"initTotalSize ${initTotalSize} " +
      s"with schema ${clippedSchema} and extraInfo ${extraInfo}")

    val myIx: Long = ix.getAndIncrement()
    q.offer(Task(
      filesAndBlocks,
      myIx,
      initTotalSize,
      writeHeaderFn,
      blockStreamerFn,
      blockCombinerFn,
      readerToTableFn,
      finalFn,
      clippedSchema,
      extraInfo))

    Some(getMySlice(myIx))
  }
}
