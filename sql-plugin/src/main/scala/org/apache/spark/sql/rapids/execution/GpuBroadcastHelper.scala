/*
 * Copyright (c) 2021-2022, NVIDIA CORPORATION.
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

package org.apache.spark.sql.rapids.execution

import ai.rapids.cudf.{NvtxColor, NvtxRange}
import com.nvidia.spark.rapids.{Arm, GpuColumnVector, SpillableColumnarBatch}
import com.nvidia.spark.rapids.RapidsBuffer
import com.nvidia.spark.rapids.LazySpillableColumnarBatch
import com.nvidia.spark.rapids.shims.SparkShimImpl
import org.apache.spark.TaskContext
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.internal.Logging
import com.nvidia.spark.rapids.GpuExpression

object GpuBroadcastHelper extends Arm with Logging {
  /**
   * Given a broadcast relation get a ColumnarBatch that can be used on the GPU.
   *
   * The broadcast relation may or may not contain any data, so we special case
   * the empty relation case (hash or identity depending on the type of join).
   *
   * If a broadcast result is unexpected we throw, but at this moment other
   * cases are not known, so this is a defensive measure.
   *
   * @param broadcastRelation - the broadcast as produced by a broadcast exchange
   * @param broadcastSchema - the broadcast schema
   * @return a `ColumnarBatch` or throw if the broadcast can't be handled
   */
  def getBroadcastBatch(broadcastRelation: Broadcast[Any],
                        broadcastSchema: StructType,
                        builtAnyNullable: Boolean,
                        boundBuiltKeys: Seq[GpuExpression]): Option[SpillableColumnarBatch] = {
    broadcastRelation.value match {
      case broadcastBatch: SerializeConcatHostBuffersDeserializeBatch =>
        Some(broadcastBatch.batch(builtAnyNullable, boundBuiltKeys))
      case v if SparkShimImpl.isEmptyRelation(v) =>
        None
        //withResource(GpuColumnVector.emptyBatch(broadcastSchema)) { emptyBatch =>
        //  val empty =
        //    LazySpillableColumnarBatch(
        //      emptyBatch,
        //      RapidsBuffer.defaultSpillCallback,
        //      "empty_built_batch")
        //  empty.allowSpilling()
        //  empty
        //}
      case t =>
        throw new IllegalStateException(s"Invalid broadcast batch received $t")
    }
  }

  /**
   * Users of this function must close the lazy batch returned after use
   * @param maybeLazyBatch - could or could not contain a lazy batch
   *                       if not, this is the signal to return an empty batch given the
   *                       schema provided.
   * @param broadcastSchema - schema to use to generate empty batches
   * @return - a lazy batch
   */
  def builtOrEmpty(maybeSpillableBatch: Option[SpillableColumnarBatch],
                   broadcastSchema: StructType): SpillableColumnarBatch = {
    maybeSpillableBatch.map { lb =>
      // we refCount++ our lazy batch, so multiple tasks are sharing this
      logWarning(s"Got lazy batch ${lb} incRefCount ${lb}")
      lb.incRefCount()
    }.getOrElse {
      withResource(GpuColumnVector.emptyBatch(broadcastSchema)) { emptyBatch =>
        val empty =
          SpillableColumnarBatch(
            emptyBatch,
            -1,
            RapidsBuffer.defaultSpillCallback)
        logWarning(s"Made empty batch ${empty}")
        empty
      }
    }
  }

  /**
   * Given a broadcast relation get the number of rows that the received batch
   * contains
   *
   * The broadcast relation may or may not contain any data, so we special case
   * the empty relation case (hash or identity depending on the type of join).
   *
   * If a broadcast result is unexpected we throw, but at this moment other
   * cases are not known, so this is a defensive measure.
   *
   * @param broadcastRelation - the broadcast as produced by a broadcast exchange
   * @return number of rows for a batch received, or 0 if it's an empty relation
   */
  def getBroadcastBatchNumRows(broadcastRelation: Broadcast[Any]): Int = {
    broadcastRelation.value match {
      case broadcastBatch: SerializeConcatHostBuffersDeserializeBatch =>
        broadcastBatch.numRows
      case v if SparkShimImpl.isEmptyRelation(v) => 0
      case t =>
        throw new IllegalStateException(s"Invalid broadcast batch received $t")
    }
  }
}

