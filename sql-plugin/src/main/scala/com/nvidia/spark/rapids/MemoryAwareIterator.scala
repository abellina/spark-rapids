package com.nvidia.spark.rapids

import ai.rapids.cudf.Table
import org.apache.spark.TaskContext
import org.apache.spark.internal.Logging

trait MemoryAwareLike extends Logging {
  def getWrapped: Any
  def getName: String

  private var _targetSize: Option[TargetSize] = None
  def getTargetSize: Option[TargetSize] = _targetSize
  def setTargetSize(newTarget: Option[TargetSize]): Unit = {
    _targetSize = newTarget
  }

  def getMemoryRequired: Long = {
    var isMemAware = true
    var current: Any = getWrapped
    val name = getName
    val targetSize = getTargetSize
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
    current = getWrapped
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
}

abstract class AbstractMemoryAwareIterator[T](
    val name: String,
    val wrapped: Any)
  extends Iterator[T]
    with MemoryAwareLike with Logging {
  def getWrapped: Any = wrapped
  def getName: String = name
  def updateMemory(context: TaskContext, table: Table, waitMetric: GpuMetric): Unit = {
    val targetSize = getTargetSize
    logInfo(s"updating memory ${name}. My memory requirement is ${targetSize}")
    if (targetSize.isDefined) {
      GpuSemaphore.updateMemory(context,
        math.ceil(targetSize.get.targetSizeBytes.toDouble/1024/1024).toInt,
        waitMetric)
    }
  }
  override def hasNext: Boolean// = wrapped.hasNext
  override def next(): T// = wrapped.next()
}

class MemoryAwareIterator[T](
  name: String,
  wrapped: Iterator[T],
  myTargetSize: Option[TargetSize] = None)
  extends AbstractMemoryAwareIterator[T](name, wrapped) {
  setTargetSize(myTargetSize)
  override def hasNext: Boolean = wrapped.hasNext
  override def next(): T = wrapped.next()
}
