package com.nvidia.spark.rapids

import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.channels.ReadableByteChannel
import java.util.concurrent.atomic.{AtomicLong, AtomicReference}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.ast.UnaryOperator
import alluxio.collections.ConcurrentHashSet
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.ScalableTaskCompletion.onTaskCompletion
import org.apache.spark.TaskContext

import org.apache.spark.internal.Logging

class MetricBucket extends AutoCloseable {
  var metric = new AtomicLong(0L)
  var refCount: Int = 0
  var pos: Int = 0
  var buckets: MetricBuckets = null
  def addMetric(m: Long): Unit = { metric.addAndGet(m) }
  def getValue: Long = { metric.get() }
  def addReference(sz: Int): Unit = synchronized { refCount += sz }
  def setPosition(i: Int, bs: MetricBuckets): Unit = {
    pos = i
    buckets = bs
  }
  override def close(): Unit = synchronized {
    refCount -= 1
    if (refCount == 0) {
      buckets.removeAt(pos)
    }
  }
}

class MetricBuckets {
  val buckets = new ArrayBuffer[MetricBucket]()
  var last = new AtomicReference[MetricBucket](new MetricBucket())
  var minBucket: Int = 0

  def addBucket(b: MetricBucket): Unit = synchronized {
    b.setPosition(buckets.size, this)
    buckets.append(b)
    last.set(buckets.last)
  }

  def removeAt(ix: Int): Unit = synchronized {
    buckets.remove(ix)
    if (buckets.nonEmpty) {
      last.set(buckets.last)
    } else {
      last.set(new MetricBucket())
    }
    minBucket = if (buckets.size == 0) {
      0
    } else {
      buckets.head.pos
    }
  }

  def addBytesToLast(bytes: Long): Unit = {
    val l = last.get()
    if (l != null) {
      l.addMetric(bytes)
    }
  }

  def closeLast(): Unit = synchronized {
    buckets.last.close()
  }

  def getSumFrom(s: Int): Long = synchronized {
    (Math.max(s, minBucket) until buckets.length).map (ix => buckets(ix).getValue).sum
  }
}

abstract class BWListener(val name: String) extends AutoCloseable {
  var bytes = 0L
  var _startingBucket: Int = 0
  val startTime = System.currentTimeMillis()
  var endTime = 0L
  var closed: Boolean = false

  def startingBucket(i: Int): Unit = {
    _startingBucket = i
  }

  def addBytesNoCallback(newBytes: Long): Unit = synchronized {
    bytes += newBytes
  }

  def getBW(): Double = synchronized {
    (bytes.toDouble / ((endTime - startTime).toDouble / 1000)) / 1024.0 / 1024.0
  }
  override def close(): Unit = synchronized {
    if (!closed) {
      closed = true
      endTime = System.currentTimeMillis()
      IOMetrics.deregister(this)
    }
  }
}

class InputBWListener(name: String, os: RegisterableStream)
    extends BWListener(name) {
  os.register(this)

  def addBytes(newBytes: Long): Unit = synchronized {
    IOMetrics.addBytesToLatest(newBytes)
  }
}

class ComputeBWListener(name: String) extends BWListener(name) {
  def addBytes(newBytes: Long): Unit = synchronized {
    IOMetrics.addBytesToLatestCompute(newBytes)
  }
}

trait RegisterableStream {
  def register(listener: InputBWListener): Unit
}

class MeteredOutStream(os: HostMemoryOutputStream)
    extends HostMemoryOutputStream(os.buffer)
        with RegisterableStream {
  var _listener: InputBWListener = null
  override def register(listener: InputBWListener): Unit = {
    _listener = listener
  }

  override def write(i: Int): Unit = {
    super.write(i)
    _listener.addBytes(1)
  }

  override def write(bytes: Array[Byte]): Unit = {
    super.write(bytes)
    _listener.addBytes(bytes.length)
  }

  override def write(bytes: Array[Byte], offset: Int, len: Int): Unit = {
    super.write(bytes, offset, len)
    _listener.addBytes(len)
  }

  override def write(data: ByteBuffer): Unit = {
    val numBytes = data.remaining()
    super.write(data)
    _listener.addBytes(numBytes)
  }

  override def copyFromChannel(channel: ReadableByteChannel, length: Long): Unit = {
    super.copyFromChannel(channel, length)
    _listener.addBytes(length)
  }
}

class MeteredRegularOutStream(os: OutputStream)
    extends OutputStream
        with RegisterableStream  {
  var _listener: InputBWListener = null

  override def register(listener: InputBWListener): Unit = {
    _listener = listener
  }

  override def write(i: Int): Unit = {
    os.write(i)
    _listener.addBytes(1)
  }
}

class CompleteListeners {
  val completeInOrder = new ArrayBuffer[BWListener]()

  def add(listener: BWListener): Unit = {
    completeInOrder.append(listener)
  }

  override def toString: String = {
    val sb = new StringBuffer()
    val l = completeInOrder.length
    completeInOrder.zipWithIndex.foreach { case (c, ix) =>
      sb.append(
        s"    ${c.name}: [" +
          s"bandwidth: ${c.getBW()} MB/sec, " +
          s"size: ${c.bytes} B, " +
          s"time: ${c.endTime - c.startTime} ms" +
        s"]\n")
      if (ix < l) {
        sb.append(", ")
      }
    }
    sb.toString
  }
}

class IOMetrics extends Logging {
  val listeners = new mutable.HashSet[BWListener]()

  private val _buckets: MetricBuckets = new MetricBuckets()
  def register(listener: BWListener): Unit = synchronized {
    listeners.add(listener)
    val newBucket = new MetricBucket()
    newBucket.addReference(listeners.size)
    listener.startingBucket(_buckets.buckets.size)
    addBucket(newBucket)
  }

  def unregister(listener: BWListener): Unit = synchronized {
    val starting = listener._startingBucket
    listener.addBytesNoCallback(_buckets.getSumFrom(starting))
    listeners.remove(listener)
    _buckets.closeLast()
  }

  def addBucket(newBucket: MetricBucket): Unit = {
    _buckets.addBucket(newBucket)
  }

  def addBytesToLatest(bytes: Long): Unit = {
    _buckets.addBytesToLast(bytes)
  }
}

object IOMetrics extends Logging {
  private var _io: IOMetrics = null
  private var _compute: IOMetrics = null
  val completePerTask = new mutable.HashMap[Long, CompleteListeners]()

  def io(): IOMetrics = {
    if (_io == null) {
      synchronized {
        if (_io == null) {
          _io = new IOMetrics()
        }
      }
    }
    _io
  }

  def compute(): IOMetrics = {
    if (_compute== null) {
      synchronized {
        if (_compute == null) {
          _compute = new IOMetrics()
        }
      }
    }
    _compute
  }

  def withInputBWMetric[T](name: String, os: RegisterableStream)(body: InputBWListener => T): T = {
    val newListener = new InputBWListener(name, os)
    io().register(newListener)
    withResource(newListener) { _ => body(newListener) }
  }

  def withComputeMetric[T](name: String)(body: ComputeBWListener => T): T = {
    val newListener = new ComputeBWListener(name)
    compute().register(newListener)
    withResource(newListener) { _ => body(newListener) }
  }

  def deregister(l: BWListener): Unit = {
    l match {
      case cbl: ComputeBWListener =>
        compute().unregister(cbl)
      case iol: InputBWListener =>
        io().unregister(iol)
      case _ =>
        throw new IllegalStateException(s"Unknown listener ${l}")
    }
    synchronized {
      val tc = TaskContext.get()
      val attempt = tc.taskAttemptId()
      if (!completePerTask.contains(attempt)) {
        onTaskCompletion(tc) {
          val completed = completePerTask.remove(attempt)
          logInfo(s"Task ${attempt} metrics:\n${completed}")
        }
        completePerTask.put(attempt, new CompleteListeners)
      }
      completePerTask(attempt).add(l)
    }
  }

  def addBytesToLatest(bytes: Long): Unit = {
    io().addBytesToLatest(bytes)
  }

  def addBytesToLatestCompute(bytes: Long): Unit = {
    compute().addBytesToLatest(bytes)
  }
}