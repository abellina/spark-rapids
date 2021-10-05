/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

package org.apache.spark.sql.rapids

import java.io.File
import java.util.UUID
import java.util.concurrent.atomic.AtomicInteger
import java.util.function.IntUnaryOperator

import com.nvidia.spark.rapids.RapidsBufferId

import org.apache.spark.storage.TempLocalBlockId

object BatchNames  {
  val AGGREGATE_OOC: String = "AggregateOOC"

  val CONCAT: String = "Concat"

  val JOIN_SPLIT: String = "Join Split"

  val SORT: String = "Sort"

  val AGGREGATE: String = "Aggregates"

  val COALESCE: String = "Coalesce"
}

object TempSpillBufferId {
  private val MAX_TABLE_ID = Integer.MAX_VALUE
  private val TABLE_ID_UPDATER = new IntUnaryOperator {
    override def applyAsInt(i: Int): Int = if (i < MAX_TABLE_ID) i + 1 else 0
  }

  /** Tracks the next table identifier */
  private[this] val tableIdCounter = new AtomicInteger(0)

  def apply(name: String): TempSpillBufferId = {
    val tableId = tableIdCounter.getAndUpdate(TABLE_ID_UPDATER)
    val tempBlockId = TempLocalBlockId(UUID.randomUUID())
    new TempSpillBufferId(tableId, tempBlockId, name)
  }
}

case class TempSpillBufferId private(
    override val tableId: Int,
    bufferId: TempLocalBlockId,
    name: String) extends RapidsBufferId {

  override def getDiskPath(diskBlockManager: RapidsDiskBlockManager): File =
    diskBlockManager.getFile(bufferId)
}