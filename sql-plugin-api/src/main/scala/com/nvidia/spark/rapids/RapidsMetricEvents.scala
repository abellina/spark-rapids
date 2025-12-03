/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.nvidia.spark.rapids

import java.nio.{ByteBuffer, ByteOrder}
import java.util.Base64

import org.apache.spark.scheduler.SparkListenerEvent

/**
 * Defines the metrics tracked for a given executor in positional order.
 * This is unshimmable API used by both the RAPIDS plugin and history server
 * to interpret metric update events.
 */
case class MetricDefinition(executorId: String, metricNames: Seq[String])
    extends SparkListenerEvent {
  override def logEvent: Boolean = true
}

/**
 * Metric updates encoded as a Base64 string of the binary payload consisting of
 * (timestamp, value) long pairs. The metric position in the sequence is defined
 * by a prior MetricDefinition.
 */
case class MetricUpdates(executorId: String, encodedMetricsBase64: String)
    extends SparkListenerEvent {
  override def logEvent: Boolean = true
}

object MetricUpdates {

  /**
   * Helper to build a MetricUpdates message from an array of (timestamp, values) tuples.
   *
   * The logical format is:
   *   Seq((timestamp0, Array(v00, v01, ...)),
   *       (timestamp1, Array(v10, v11, ...)), ...)
   *
   * The binary layout is:
   *   [int numUpdates]
   *   [long ts0][long v00][long v01]...
   *   [long ts1][long v10][long v11]...
   *   ...
   * All integers are big-endian. Each values array is assumed to be the same length.
   */
  def fromArrays(
      executorId: String,
      updates: Array[(Long, Array[Long])]): MetricUpdates = {
    val numUpdates = updates.length
    val numMetricsPerUpdate =
      if (numUpdates == 0) 0 else updates(0)._2.length

    // allocate space for count + all updates
    val bb = ByteBuffer
      .allocate(
        Integer.BYTES + // numUpdates
          numUpdates * (java.lang.Long.BYTES + numMetricsPerUpdate * java.lang.Long.BYTES))
      .order(ByteOrder.BIG_ENDIAN)

    // prefix with number of updates
    bb.putInt(numUpdates)

    // encode each (timestamp, Array[Long]) consecutively
    var i = 0
    while (i < numUpdates) {
      val (ts, values) = updates(i)
      bb.putLong(ts)
      var j = 0
      while (j < numMetricsPerUpdate) {
        bb.putLong(values(j))
        j += 1
      }
      i += 1
    }

    MetricUpdates(executorId, Base64.getEncoder.encodeToString(bb.array()))
  }
}


