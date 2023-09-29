package com.nvidia.spark.rapids

import java.nio.ByteBuffer
import java.nio.channels.ReadableByteChannel

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer

class MetricBucket extends AutoCloseable {
  var metric: Long = 0L
  var refCount: Int = 0
  var pos: Int = 0
  var buckets: MetricBuckets = null
  def addMetric(m: Long): Unit = synchronized { metric += m }
  def getValue: Long = synchronized { metric }
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
  var minBucket: Int = 0

  def addBucket(b: MetricBucket): Unit = synchronized {
    b.setPosition(buckets.size, this)
    buckets.append(b)
  }

  def removeAt(ix: Int): Unit = synchronized {
    buckets.remove(ix)
    minBucket = buckets.head.pos
  }

  def addBytesToLast(bytes: Long): Unit = synchronized {
    buckets.last.addMetric(bytes)
  }

  def closeLast(): Unit = synchronized {
    buckets.last.close()
  }

  def getSumFrom(s: Int): Long = synchronized {
    (Math.max(s, minBucket) until buckets.length).map (ix => buckets(ix).getValue).sum
  }
}

trait BWListener extends AutoCloseable {
  var bytes = 0L
  var _startingBucket: Int = 0
  val startTime = System.currentTimeMillis()
  var endTime = 0L

  def startingBucket(i: Int): Unit = {
    _startingBucket = i
  }

  def addBytesNoCallback(newBytes: Long): Unit = synchronized {
    bytes += newBytes
  }

  def getBW(): Double = synchronized {
    (bytes.toDouble / ((endTime - startTime).toDouble / 1000)) / 1024.0 / 1024.0
  }
}

class InputBWListener(os: MeteredOutStream) extends BWListener {
  var closed = false
  os.register(this)

  override def close(): Unit = {
    if (!closed) {
      closed = true
      endTime = System.currentTimeMillis()
      IOMetrics.deregister(this)
    }
  }

  def addBytes(newBytes: Long): Unit = {
    IOMetrics.addBytesToLatest(newBytes)
  }
}

class ComputeBWListener extends BWListener {
  var closed = false
  override def close(): Unit = {
    if (!closed) {
      closed = true
      endTime = System.currentTimeMillis()
      IOMetrics.deregister(this)
    }
  }

  def addBytes(newBytes: Long): Unit = {
    IOMetrics.addBytesToLatestCompute(newBytes)
  }
}

class MeteredOutStream(os: HostMemoryOutputStream)
    extends HostMemoryOutputStream(os.buffer) {
  var _listener: InputBWListener = null
  def register(listener: InputBWListener): Unit = {
    _listener = listener
  }

  override def write(i: Int): Unit = {
    super.write(i)
    _listener.addBytes(4)
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

class IOMetrics {
  val listeners = new mutable.HashSet[BWListener]()
  private val _buckets: MetricBuckets = new MetricBuckets()
  def register(listener: BWListener): Unit = synchronized {
    listeners.add(listener)
    val newBucket = new MetricBucket()
    newBucket.addReference(listeners.size)
    listener.startingBucket(_buckets.buckets.size)
    addBucket(newBucket)
    println(s"registered ${listeners}")
  }

  def unregister(listener: BWListener): Unit = synchronized {
    val starting = listener._startingBucket
    listener.addBytesNoCallback(_buckets.getSumFrom(starting))
    listeners.remove(listener)
    _buckets.closeLast()
    println(s"unregistered ${listeners}")
  }

  def addBucket(newBucket: MetricBucket): Unit = synchronized {
    _buckets.addBucket(newBucket)
  }

  def addBytesToLatest(bytes: Long): Unit = synchronized {
    _buckets.addBytesToLast(bytes)
  }
}

object IOMetrics {
  private var _io: IOMetrics = null
  private var _compute: IOMetrics = null
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

  def registerInputBW(os: MeteredOutStream): InputBWListener = {
    val newListener = new InputBWListener(os)
    io().register(newListener)
    newListener
  }

  def computeMetric(): ComputeBWListener = {
    val newListener = new ComputeBWListener()
    compute().register(newListener)
    newListener
  }

  def deregister(l: InputBWListener): Unit = {
    io().unregister(l)
  }

  def deregister(l: ComputeBWListener): Unit = {
    compute().unregister(l)
  }

  def addBytesToLatest(bytes: Long): Unit = {
    io().addBytesToLatest(bytes)
  }

  def addBytesToLatestCompute(bytes: Long): Unit = {
    compute().addBytesToLatest(bytes)
  }
}