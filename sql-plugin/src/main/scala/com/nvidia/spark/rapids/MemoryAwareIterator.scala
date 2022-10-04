package com.nvidia.spark.rapids

import ai.rapids.cudf.Table
import org.apache.spark.{InterruptibleIterator, TaskContext}
import org.apache.spark.internal.Logging

import scala.collection.mutable.ArrayBuffer

trait MemoryAwareLike extends Logging {
  def getWrapped: Any
  def getName: String

  private var _targetSize: Option[TargetSize] = None
  def getTargetSize: Option[TargetSize] = _targetSize
  def setTargetSize(newTarget: Option[TargetSize]): Unit = {
    _targetSize = newTarget
  }

  var child: MemoryAwareLike = null

  install

  def install: Unit = {
    var isMemAware = true
    var current: Any = getWrapped
    val name = getName
    val targetSize = getTargetSize
    logInfo(s"${TaskContext.get.taskAttemptId()}: starting at ${name}. ${this.getClass}. " +
      s"target: ${getTargetSize}")
    var theTarget: Option[TargetSize] = None
    val stack = new ArrayBuffer[MemoryAwareLike]()
    stack.append(this)
    while (isMemAware) {
      current match {
        case i: InterruptibleIterator[_] =>
          current = i.delegate
        case mai: MemoryAwareLike =>
          if (mai.getTargetSize.isDefined) {
            if (theTarget.isEmpty ||
              theTarget.get.targetSizeBytes < mai.getTargetSize.get.targetSizeBytes) {
              theTarget = mai.getTargetSize
            }
          }
          current = mai.getWrapped
          stack.append(mai)
        case null => isMemAware = false// at end
        case _ =>
          isMemAware = false
      }
    }
    (0 until stack.length - 1).foreach { ix =>
      stack(ix + 1).child = stack(ix)
    }
  }

  def getMemoryRequired: Long = {
    var targetSize: Option[TargetSize] = getTargetSize
    val ctx = TaskContext.get()
    var currentChild = child
    while (currentChild != null) {
      logInfo(s"Finding requirement =>  " +
        s"starting node: ${getName} " +
        s"task: ${ctx.taskAttemptId()} " +
        s"at child: ${currentChild.getName} " +
        s"child target: ${currentChild.getTargetSize}")
      if (currentChild.getTargetSize.isDefined) {
        if (targetSize.isEmpty) {
          targetSize = currentChild.getTargetSize
        } else {
          if (currentChild.getTargetSize.get.targetSizeBytes >
                targetSize.get.targetSizeBytes){
            targetSize = currentChild.getTargetSize
          }
        }
      }
      currentChild = currentChild.child
    }
    val memReq = targetSize.map(_.targetSizeBytes).getOrElse(-1L)

    logInfo(s"Memory requirement =>  " +
      s"node: ${getName} " +
      s"task: ${ctx.taskAttemptId()} " +
      s"is: ${memReq}")
    memReq
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
