/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION.
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

import ai.rapids.cudf.{ColumnVector, DType, Table}
import ai.rapids.cudf.HostColumnVector.{BasicType, ListType}

import java.util
import org.apache.spark.sql.types.{ArrayType, DataType, IntegerType}
import org.apache.spark.sql.vectorized.ColumnarBatch

class GpuGenerateSuite extends SparkQueryCompareTestSuite {
  val rapidsConf = new RapidsConf(Map.empty[String, String])

  def makeBatch(includeSecondColumn: Boolean): ColumnarBatch = {
    val list = util.Arrays.asList(1, 2, 3, 4)
    val rows = (0 until 100).map(r => list)
    val cvList = ColumnVector.fromLists(
      new ListType(true,
        new BasicType(true, DType.INT32)),
      rows: _*)

    if (includeSecondColumn) {
      val dt: Array[DataType] = Seq(IntegerType, ArrayType(IntegerType)).toArray
      GpuColumnVector.from(
        new Table(ColumnVector.fromInts((0 until 100):_*), cvList), dt)
    } else {
      val dt: Array[DataType] = Seq(ArrayType(IntegerType)).toArray
      GpuColumnVector.from(new Table(cvList), dt)
    }
  }

  test("test batch with a single exploding column") {
    val batch = makeBatch(includeSecondColumn = false)
    val e = GpuExplode(null)
    // the exploded column should be 4 Bytes * 100 rows * 4 reps per row = 1600 Bytes.
    // 1600 == a single split
    assertResult(0)(
      e.inputSplitIndices(batch, 0, false, 1600).length)

    // 800 == 1 splits (2 parts) right down the middle
    var splits = e.inputSplitIndices(batch, 0, false, 800)
    assertResult(1)(splits.length)
    assertResult(50)(splits(0))

    // 400 == 3 splits (4 parts)
    splits = e.inputSplitIndices(batch, 0, false, 400)
    assertResult(3)(splits.length)
    assertResult(25)(splits(0))
    assertResult(50)(splits(1))
    assertResult(75)(splits(2))
  }

  test("test batch with a two a repeating column") {
    val batch = makeBatch(includeSecondColumn = true)
    val e = GpuExplode(null)
    // the exploded column should be 4 Bytes * 100 rows * 4 reps per row = 1600 Bytes.
    // the repeating column is 4 bytes (or 400 bytes total) repeated 4 times (or 1600)
    // 1600 == a single split
    assertResult(1)(
      e.inputSplitIndices(batch, 1, false, 1600).length)

    // 1600 == 1 splits (2 parts) right down the middle
    var splits = e.inputSplitIndices(batch, 1, false, 800)
    assertResult(3)(splits.length)
    assertResult(24)(splits(0))
    assertResult(49)(splits(1))
    assertResult(74)(splits(2))

    // 800 == 3 splits (4 parts)
    splits = e.inputSplitIndices(batch, 1, false, 400)
    assertResult(7)(splits.length)
    assertResult(12)(splits(0))
    assertResult(24)(splits(1))
    assertResult(37)(splits(2))
    assertResult(49)(splits(3))
    assertResult(62)(splits(4))
    assertResult(74)(splits(5))
    assertResult(87)(splits(6))

    splits = e.inputSplitIndices(batch, 1, false, 100)
    // every 3
    assertResult(3)(splits.length)
    assertResult(25)(splits(0))
    assertResult(50)(splits(1))
    assertResult(75)(splits(2))
  }
}
