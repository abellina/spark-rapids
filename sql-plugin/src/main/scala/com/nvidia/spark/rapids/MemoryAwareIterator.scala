package com.nvidia.spark.rapids

import ai.rapids.cudf.Table
import org.apache.spark.TaskContext

trait MemoryAwareLike {
  def getWrapped: Any
  def getMemoryRequired: Long
  def getName: String
  def getTargetSize: Option[TargetSize]
}

abstract class AbstractMemoryAwareIterator[T](
    val name: String,
    val wrapped: Any) extends Iterator[T] with MemoryAwareLike {
  def getWrapped: Any = wrapped

  def getName: String = name

  def updateMemory(context: TaskContext, table: Table, waitMetric: GpuMetric): Unit = {
    GpuSemaphore.updateMemory(context, table, waitMetric)
  }

  var myRequiredMemory: Option[TargetSize] = None

  def getMemoryRequired: Long = {
    var isMemAware = true
    var current: Any = wrapped
    if (current == null) {
      println(s"${TaskContext.get.taskAttemptId()}: starting at ${name}. ${this.getClass}. " +
        s"NOTHING WRAPPED")
      return 0L
    }

    println(s"${TaskContext.get.taskAttemptId()}: starting at ${name}. ${this.getClass}. " +
      s"target: ${getTargetSize}")
    while (isMemAware) {
      current match {
        case mai: MemoryAwareLike =>
          println(s"${TaskContext.get.taskAttemptId()}: $current is MemoryAwareIter. " +
            s"${mai.getClass}: ${mai.getName}, target: ${getTargetSize}")
          if (mai.getTargetSize.isDefined) {
            myRequiredMemory = mai.getTargetSize
          }
          current = mai.getWrapped
        case _ =>
          println(s"${TaskContext.get.taskAttemptId()}: $current is NOT MemoryAwareIter, " +
            s"it is $current, ${current.getClass}, target: ${getTargetSize}")
          isMemAware = false
      }
    }
    println(s"my required size $this is ${myRequiredMemory}")
    myRequiredMemory.map(_.targetSizeBytes).getOrElse(-1L)
  }

  override def hasNext: Boolean// = wrapped.hasNext
  override def next(): T// = wrapped.next()
}

class MemoryAwareIterator[T](
  name: String,
  wrapped: Iterator[T],
  targetSize: Option[TargetSize] = None)
  extends AbstractMemoryAwareIterator[T](name, wrapped) {
  override def hasNext: Boolean = wrapped.hasNext
  override def next(): T = wrapped.next()
  override def getTargetSize: Option[TargetSize] = targetSize
}
