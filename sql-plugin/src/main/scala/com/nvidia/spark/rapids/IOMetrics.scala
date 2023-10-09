package com.nvidia.spark.rapids

import java.io.OutputStream
import java.nio.ByteBuffer
import java.nio.channels.ReadableByteChannel
import java.util.concurrent.atomic.{AtomicLong, AtomicReference}
import java.util.concurrent.{ConcurrentHashMap, Executor, Executors, TimeUnit}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

import ai.rapids.cudf.ast.UnaryOperator
import alluxio.collections.ConcurrentHashSet
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.ScalableTaskCompletion.onTaskCompletion
import com.nvidia.spark.rapids.jni.RmmSpark
import org.apache.spark.TaskContext

import org.apache.spark.internal.Logging

class MetricBucket extends AutoCloseable {
  var metric = new AtomicLong(0L)
  var refCount: Int = 0
  var pos: Int = 0
  var buckets: MetricBuckets = null
  var count = 0
  def addMetric(m: Long): Unit = { metric.addAndGet(m) }
  def getValue: Long = { metric.get() }
  def getCount: Long = { count }
  def addReference(sz: Int): Unit = synchronized {
    count = count + 1
    refCount += sz
  }
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

class MetricBuckets extends Logging {
  val buckets = new ArrayBuffer[MetricBucket]()
  var last = new AtomicReference[MetricBucket](null)
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
      last.set(null)
    }
    minBucket = if (buckets.isEmpty) {
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

  def getSumFrom(s: Int): (Long, Long) = synchronized {
    val bs = (Math.max(s, minBucket) until buckets.length).map(ix => buckets(ix))
    (bs.map(_.getValue).sum, bs.map(_.getCount).sum)
  }
}

abstract class BWListener(val name: String, val tc: TaskContext) extends AutoCloseable {
  var bytes = 0L
  var myBytes = 0L
  var _startingBucket: Int = 0
  val startTime = System.nanoTime()
  var endTime = 0L
  var count = 0L
  var closed: Boolean = false

  def startingBucket(i: Int): Unit = {
    _startingBucket = i
  }

  def addBytesNoCallback(newBytes: Long, c: Long): Unit = synchronized {
    bytes += newBytes
    count += c
  }

  def addBytes(newBytes: Long): Unit

  def getBW(): Double = synchronized {
    (bytes.toDouble / ((endTime - startTime).toDouble / 1000000000.0D)) / 1024.0 / 1024.0
  }

  def getMyBW(): Double = synchronized {
    (myBytes.toDouble / ((endTime - startTime).toDouble / 1000000000.0D)) / 1024.0 / 1024.0
  }

  override def close(): Unit = synchronized {
    if (!closed) {
      closed = true
      endTime = System.nanoTime()
      IOMetrics.deregister(this)
    }
  }
}

class InputBWListener(name: String, os: RegisterableStream, tc: TaskContext)
    extends BWListener(name, tc) {
  os.register(this)

  override def addBytes(newBytes: Long): Unit = synchronized {
    myBytes += newBytes
    IOMetrics.addBytesToLatest(name, newBytes)
  }
}

class BWListenerPerTask(name: String, tc: TaskContext)
    extends BWListener(name, tc) {

  override def addBytes(newBytes: Long): Unit = synchronized {
    myBytes += newBytes
    IOMetrics.addBytesToLatest(name, newBytes)
  }
}

class ComputeBWListener(name: String, tc: TaskContext) extends BWListener(name, tc) {
  override def addBytes(newBytes: Long): Unit = synchronized {
    myBytes += newBytes
    IOMetrics.addBytesToLatestCompute(name, newBytes)
  }
}

trait RegisterableStream {
  def register(listener: BWListener): Unit
}

class MeteredOutStream(os: HostMemoryOutputStream)
    extends HostMemoryOutputStream(os.buffer)
        with RegisterableStream {
  var _listeners = new ArrayBuffer[BWListener]()
  override def register(listener: BWListener): Unit = {
    _listeners.append(listener)
  }

  override def write(i: Int): Unit = {
    super.write(i)
    _listeners.foreach(_.addBytes(1))
  }

  override def write(bytes: Array[Byte]): Unit = {
    super.write(bytes)
    _listeners.foreach(_.addBytes(bytes.length))
  }

  override def write(bytes: Array[Byte], offset: Int, len: Int): Unit = {
    super.write(bytes, offset, len)
    _listeners.foreach(_.addBytes(len))
  }

  override def write(data: ByteBuffer): Unit = {
    val numBytes = data.remaining()
    super.write(data)
    _listeners.foreach(_.addBytes(numBytes))
  }

  override def copyFromChannel(channel: ReadableByteChannel, length: Long): Unit = {
    super.copyFromChannel(channel, length)
    _listeners.foreach(_.addBytes(length))
  }
}

class MeteredRegularOutStream(os: OutputStream)
    extends OutputStream
        with RegisterableStream  {
  var _listener: BWListener = null

  override def register(listener: BWListener): Unit = {
    _listener = listener
  }

  override def write(i: Int): Unit = {
    os.write(i)
    _listener.addBytes(1)
  }
}

class CompleteListeners {
  var listener: BWListenerPerTask = null
  val completeInOrder = new ArrayBuffer[BWListener]()

  def add(listener: BWListener): Unit = {
    require(listener.closed)
    completeInOrder.append(listener)
  }

  class StatBucket(val bucketMaxSize: Long) {
    var count: Long = 0
    var sum: Double = 0
    var min: Double = Double.MaxValue
    var max: Double = 0

    def update(listener: BWListener): Unit = {
      val bw = listener.getBW()
      if (bw < min) {
        min = bw
      }
      if (bw > max) {
        max = bw
      }
      sum += bw
      count += listener.count
    }

    override def toString(): String = {
      val sb = new StringBuilder()
      val mean = sum/count
      sb.append(s"'$bucketMaxSize': {'min': $min, 'max': $max, 'mean': $mean, 'count': $count}")
      sb.toString
    }
  }

  class Stats {
    var min: Double = Double.MaxValue
    var max: Double = 0
    var sum: Double = 0
    var count: Long = 0
    val buckets = new Array[StatBucket](13)
    (0 until 13).foreach { i =>
      val size = if (i == 0) {
        1024L
      } else if (i == 1) {
        1024L * 10
      } else if (i == 2) {
        1024L * 100
      } else if (i == 3) {
        1024L * 500
      } else if (i == 4) {
        1024L * 1024
      } else if (i == 5) {
        1024L * 1024 * 10
      } else if (i == 6) {
        1024L * 1024 * 50
      } else if (i == 7) {
        1024L * 1024 * 100
      } else if (i == 8) {
        1024L * 1024 * 500
      } else if (i == 9) {
        1024L * 1024 * 1024
      } else if (i == 10) {
        1024L * 1024 * 1024 * 2
      } else if (i == 11) {
        1024L * 1024 * 1024 * 5
      } else {
        // greater than 5GB!
        Long.MaxValue
      }
      buckets(i) = new StatBucket(size)
    }

    def getNearestBucket(size: Long): StatBucket = {
      val ix = if (size < 1024) {
        0
      } else if (size < 1024L * 10) {
        1
      } else if (size < 1024L * 100) {
        2
      } else if (size < 1024L * 500) {
        3
      } else if (size < 1024L * 1024) {
        4
      } else if (size < 1024L * 1024 * 10) {
        5
      } else if (size < 1024L * 1024 * 50) {
        6
      } else if (size < 1024L * 1024 * 100) {
        7
      } else if (size < 1024L * 1024 * 500) {
        8
      } else if (size < 1024L * 1024 * 1024) {
        9
      } else if (size < 1024L * 1024 * 1024 * 2) {
        10
      } else if (size < 1024L * 1024 * 1024 * 5) {
        11
      } else {
        // greater than 5GB!
        12
      }
      buckets(ix)
    }

    def update(listener: BWListener): Unit = {
      val bw = listener.getBW()
      if (bw < min) {
        min = bw
      }
      if (bw > max) {
        max = bw
      }
      sum += bw
      count += listener.count
      val b = getNearestBucket(listener.bytes)
      b.update(listener)
    }

    def printBuckets: String = {
      val sb = new StringBuilder()
      buckets.foreach { b =>
        if (b.count > 0) {
          sb.append("{")
          sb.append(b.toString + "},")
        }
      }
      sb.toString()
    }

    override def toString: String = {
      val mean = sum/count
      s"{'overall': {'min': $min, 'max': $max, 'mean': $mean, 'count': $count}, " +
          s"'bucketed': [$printBuckets]}"
    }
  }

  override def toString: String = {
    val l = completeInOrder.length
    val stats = new mutable.HashMap[String, Stats]()
    completeInOrder.foreach { c =>
      if (!stats.contains(c.name)) {
        stats.put(c.name, new Stats)
      }
      stats(c.name).update(c)
    }
    val sb = new StringBuffer()
    stats.keys.foreach { key =>
      sb.append(
        s"'${key}': ['stats': ${stats(key)}], \n")
    }
    sb.toString
  }

  var registered: Boolean = false
}

class IOMetrics extends Logging {

  val listeners = new mutable.HashSet[BWListener]()

  private val _buckets: ConcurrentHashMap[String, MetricBuckets] =
    new ConcurrentHashMap[String, MetricBuckets]()

  def register(listener: BWListener): Unit = synchronized {
    listeners.add(listener)
    val newBucket = new MetricBucket()
    val myBuckets = _buckets.computeIfAbsent(listener.name, _ => {
      new MetricBuckets()
    })
    newBucket.addReference(listeners.size)
    listener.startingBucket(myBuckets.buckets.size)
    myBuckets.addBucket(newBucket)
  }

  def unregister(listener: BWListener): Unit = synchronized {
    val starting = listener._startingBucket
    val myBuckets = _buckets.get(listener.name)
    val (sum, count) = myBuckets.getSumFrom(starting)
    listener.addBytesNoCallback(sum, count)
    listeners.remove(listener)
    myBuckets.closeLast()
  }

  def addBytesToLatest(name: String, bytes: Long): Unit = {
    val myBuckets = _buckets.get(name)
    myBuckets.addBytesToLast(bytes)
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

  def withInputBWMetric[T](
      name: String,
      os: RegisterableStream,
      tc: TaskContext = TaskContext.get)(body: InputBWListener => T): T = {
    synchronized {
      var perTask: CompleteListeners = null
      if (!completePerTask.contains(tc.taskAttemptId())) {
        perTask = new CompleteListeners
        perTask.listener = new BWListenerPerTask("taskInputBW", tc)
        io().register(perTask.listener)
        completePerTask.put(tc.taskAttemptId(), perTask)
      } else {
        perTask = completePerTask(tc.taskAttemptId())
      }
      os.register(perTask.listener)
    }
    val newListener = new InputBWListener(name, os, tc)
    io().register(newListener)
    withResource(newListener) { _ => body(newListener) }
  }

  def withOutputBWMetric[T](
      str: String, tc: TaskContext = TaskContext.get())(body: ComputeBWListener => T): T = {
    val newListener = new ComputeBWListener(str, tc)
    compute().register(newListener)
    withResource(newListener) { _ => body(newListener) }
  }

  def withComputeMetric[T](
      name: String, tc: TaskContext = TaskContext.get())(body: ComputeBWListener => T): T = {
    val newListener = new ComputeBWListener(name, tc)
    compute().register(newListener)
    withResource(newListener) { _ => body(newListener) }
  }

  def deregister(l: BWListener): Unit = {
    l match {
      case cbl: ComputeBWListener =>
        compute().unregister(cbl)
      case iol: InputBWListener =>
        io().unregister(iol)
      case pertask: BWListenerPerTask =>
        io().unregister(pertask)
      case _ =>
        throw new IllegalStateException(s"Unknown listener ${l}")
    }
    synchronized {
      val tc = l.tc
      val attempt = tc.taskAttemptId()
      val addCallback = !completePerTask.contains(attempt) || !completePerTask(attempt).registered
      if (addCallback) {
        onTaskCompletion(tc) {
          val completed = completePerTask.remove(attempt)
          val (taskBW, myTaskBW) = completed.map { c =>
            if (c.listener != null) {
              c.listener.close()
              (c.listener.getBW(), c.listener.getMyBW())
            } else  {
              (-1, -1)
            }
          }.getOrElse((-1, -1))
          logInfo(s"Task ${attempt}. System task bw: ${taskBW} MB/sec. My task bw: ${myTaskBW} " +
            s"Metrics:\n${completed}")
        }
        if (!completePerTask.contains(attempt)) {
          completePerTask.put(attempt, new CompleteListeners)
        }
        completePerTask(attempt).registered = true
      }
      completePerTask(attempt).add(l)
    }
  }

  def addBytesToLatest(name: String, bytes: Long): Unit = {
    io().addBytesToLatest(name, bytes)
  }

  def addBytesToLatestCompute(name: String, bytes: Long): Unit = {
    compute().addBytesToLatest(name, bytes)
  }
}