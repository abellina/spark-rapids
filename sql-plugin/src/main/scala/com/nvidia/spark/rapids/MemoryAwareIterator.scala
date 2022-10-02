package com.nvidia.spark.rapids

import org.apache.spark.TaskContext

trait MemoryAwareLike {
  def getWrapped: Iterator[_]
  def getMemoryRequired: Long
  def getName: String
}

abstract class AbstractMemoryAwareIterator[T, V](
    val name: String,
    val wrapped: Iterator[T]) extends Iterator[V] with MemoryAwareLike {
  def getWrapped: Iterator[_] = wrapped

  def getName: String = name

  def getMemoryRequired: Long = {
    var isMemAware = true
    var current: Any = wrapped
    if (current == null) {
      println(s"${TaskContext.get.taskAttemptId()}: starting at ${name}. ${this.getClass}. " +
        s"NOTHING WRAPPED")
      return 0L
    }

    println(s"${TaskContext.get.taskAttemptId()}: starting at ${name}. ${this.getClass}")
    while (isMemAware) {
      current match {
        case mai: MemoryAwareLike =>
          println(s"${TaskContext.get.taskAttemptId()}: $current is MemoryAwareIter. " +
            s"${mai.getClass}: ${mai.getName}")
          current = mai.getWrapped
        case _ =>
          println(s"${TaskContext.get.taskAttemptId()}: $current is NOT MemoryAwareIter, " +
            s"it is $current, ${current.getClass}")
          isMemAware = false
      }
    }
    0L
  }

  override def hasNext: Boolean// = wrapped.hasNext
  override def next(): V// = wrapped.next()
}

class MemoryAwareIterator[T](
  name: String,
  wrapped: Iterator[T]) extends AbstractMemoryAwareIterator[T, T](name, wrapped) {
  override def hasNext: Boolean = wrapped.hasNext
  override def next(): T = wrapped.next()
}
