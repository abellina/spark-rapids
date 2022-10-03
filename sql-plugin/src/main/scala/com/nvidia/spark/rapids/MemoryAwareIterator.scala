package com.nvidia.spark.rapids

import ai.rapids.cudf.Table
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging

trait MemoryAwareLike {
  def getWrapped: Any
  def getMemoryRequired: Long
  def getName: String
  def getTargetSize: Option[TargetSize]
  def setTargetSize(theTarget: Option[TargetSize]): Unit
}

abstract class AbstractMemoryAwareIterator[T](
    val name: String,
    val wrapped: Any,
    var targetSize: Option[TargetSize] = None)
  extends Iterator[T]
    with MemoryAwareLike with Logging {

  override def setTargetSize(newTarget: Option[TargetSize]): Unit = {
    targetSize = newTarget
  }

  def getWrapped: Any = wrapped

  def getName: String = name

  def updateMemory(context: TaskContext, table: Table, waitMetric: GpuMetric): Unit = {
    logInfo(s"updating memory ${name}. My memory requirement is ${targetSize}")
    if (targetSize.isDefined) {
      GpuSemaphore.updateMemory(context,
        math.ceil(targetSize.get.targetSizeBytes.toDouble/1024/1024).toInt,
        waitMetric)
    }
  }

  def getMemoryRequired: Long = {
    var isMemAware = true
    var current: Any = wrapped
    if (current == null) {
      logInfo(s"${TaskContext.get.taskAttemptId()}: starting at ${name}. ${this.getClass}. " +
        s"NOTHING WRAPPED")
      return 0L
    }

    logInfo(s"${TaskContext.get.taskAttemptId()}: starting at ${name}. ${this.getClass}. " +
      s"target: ${getTargetSize}")
    var theTarget: Option[TargetSize] = None
    while (isMemAware) {
      current match {
        case mai: MemoryAwareLike =>
          logInfo(s"${TaskContext.get.taskAttemptId()}: $current is MemoryAwareIter. " +
            s"${mai.getClass}: ${mai.getName}, target: ${getTargetSize}")
          if (mai.getTargetSize.isDefined) {
            if (theTarget.isEmpty ||
              theTarget.get.targetSizeBytes < mai.getTargetSize.get.targetSizeBytes) {
              theTarget = mai.getTargetSize
            }
          }
          current = mai.getWrapped
        case _ =>
          logInfo(s"${TaskContext.get.taskAttemptId()}: $current is NOT MemoryAwareIter, " +
            s"it is $current, ${current.getClass}, target: ${getTargetSize}")
          isMemAware = false
      }
    }
    logInfo(s"my required size $this is ${targetSize}")
    current = wrapped
    logInfo(s"${TaskContext.get.taskAttemptId()}: starting AGAIN at ${name}. ${this.getClass}. " +
      s"target: ${getTargetSize}")
    isMemAware = true
    while (isMemAware) {
      current match {
        case mai: MemoryAwareLike =>
          mai.setTargetSize(theTarget)
          current = mai.getWrapped
        case _ =>
          isMemAware = false
      }
    }
    targetSize.map(_.targetSizeBytes).getOrElse(-1L)
  }

  override def hasNext: Boolean// = wrapped.hasNext
  override def next(): T// = wrapped.next()
}

class MemoryAwareIterator[T](
  name: String,
  wrapped: Iterator[T],
  targetSize: Option[TargetSize] = None)
  extends AbstractMemoryAwareIterator[T](name, wrapped, targetSize) {
  override def hasNext: Boolean = wrapped.hasNext
  override def next(): T = wrapped.next()
  override def getTargetSize: Option[TargetSize] = targetSize
}
