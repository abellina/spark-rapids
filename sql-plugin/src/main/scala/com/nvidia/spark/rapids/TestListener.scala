package com.nvidia.spark.rapids

import org.apache.spark.executor.TaskMetrics
import org.apache.spark.scheduler.{SparkListener, SparkListenerJobStart, SparkListenerTaskEnd, SparkListenerTaskStart}

case class Stats(sum: Long, min: Long, max: Long, count: Long) {
  override def toString: String = {
    val avg = sum.toDouble/count
    s"[$count, $min, $avg, $max]"
  }
}

object TestListener {
  var theMap = scala.collection.mutable.HashMap[
    Int, scala.collection.mutable.HashMap[String, Stats]]()

  def reset(): Unit = {
    println(s"resetting map that had ${theMap.keySet.size} jobs")
    theMap.clear()
  }

  def combine(): Option[scala.collection.mutable.Map[String, Stats]] = {
    val jobs = theMap.keysIterator
    var finalStats: scala.collection.mutable.Map[String, Stats] = null
    if (jobs.hasNext) {
      while (jobs.hasNext) {
        val stats: scala.collection.mutable.Map[String, Stats] = theMap(jobs.next())
        if (finalStats == null) {
          finalStats = stats
        } else {
          val keys = stats.keysIterator
          while (keys.hasNext) {
            val key = keys.next()
            val ek = finalStats(key)
            val v = stats(key)
            finalStats.put(
              key, Stats(ek.sum + v.sum, ek.min.min(v.min), ek.max.max(v.max), ek.count + v.count))
          }
        }
      }
      Some(finalStats)
    } else {
      None
    }
  }

  def getSummary(): String = {
    val stats = combine()
    if (stats.isEmpty) {
      "{}"
    } else {
      val s = stats.get
      val metricsFormated = StringBuilder.newBuilder
      metricsFormated.append("{")
      val keys = s.keys.iterator
      var iteratedOnce = false
      while (keys.hasNext) {
        val k = keys.next()
        if (iteratedOnce) {
          metricsFormated.append(",")
        }
        metricsFormated.append(s"${k.mkString("\"", "", "\"")}: ${s(k)}")
        iteratedOnce = true
      }
      metricsFormated.append("}")
      metricsFormated.toString
    }
  }
}

class TestListener extends SparkListener
{

  override def onJobStart(jobStart: SparkListenerJobStart): Unit = {
    currentJob = jobStart.jobId
  }

  override def onTaskStart(taskStart: SparkListenerTaskStart): Unit = {
    super.onTaskStart(taskStart)
  }

  def update(jobId: Int)(key: String, v: Long): String = {
    val existing = TestListener.theMap.get(jobId)
    if (existing.isEmpty) {
      val jobMap = scala.collection.mutable.HashMap[String, Stats]()
      jobMap.put(key, Stats(v, v, v, 1))
      TestListener.theMap.put(jobId, jobMap)
    } else {
      val jobMap = TestListener.theMap(jobId)
      if (!jobMap.contains(key)) {
        jobMap.put(key, Stats(v, v, v, 1))
      } else {
        val ek = jobMap(key)
        jobMap.put(key,
          Stats(ek.sum + v, ek.min.min(v), ek.max.max(v), ek.count + 1))
      }
    }
    val stats = TestListener.theMap(jobId)(key)
    s"$jobId: $key: $stats"
  }

  var currentJob: Int = 0

  override def onTaskEnd(taskEnd: SparkListenerTaskEnd): Unit = {
    super.onTaskEnd(taskEnd)
    val tm: TaskMetrics = taskEnd.taskMetrics
    def up: (String, Long) => String = update(currentJob)(_, _)
    up("inputMetrics.bytesRead",     tm.inputMetrics.bytesRead)
    up("executor.deserTime",         tm.executorDeserializeTime)
    up("executor.deserCPUTime",      tm.executorDeserializeCpuTime)
    up("executor.runTime",           tm.executorRunTime)
    up("executor.cpuTime",           tm.executorCpuTime)
    up("executor.jvmGCTime",         tm.jvmGCTime)
    up("executor.resultSerTime",     tm.resultSerializationTime)
    up("executor.memorySpilled",     tm.memoryBytesSpilled)
    up("executor.diskSpilled",       tm.diskBytesSpilled)
    up("outputMetrics.bytesWritten", tm.outputMetrics.bytesWritten)
    up("resultSize",                 tm.resultSize)
    up("shuffleRead.remoteBlocksFetched", tm.shuffleReadMetrics.remoteBlocksFetched)
    up("shuffleRead.remoteBytesFetched", tm.shuffleReadMetrics.remoteBytesRead)
    up("shuffleRead.localBlocksFetched", tm.shuffleReadMetrics.localBlocksFetched)
    up("shuffleRead.localBytesFetched", tm.shuffleReadMetrics.localBytesRead)
    up("shuffleRead.fetchWait", tm.shuffleReadMetrics.fetchWaitTime)
    up("shuffleWriteMetrics.bytesWritten", tm.shuffleWriteMetrics.bytesWritten)
    up("shuffleReadMetrics.totalBytesRead", tm.shuffleReadMetrics.totalBytesRead)
  }
}
