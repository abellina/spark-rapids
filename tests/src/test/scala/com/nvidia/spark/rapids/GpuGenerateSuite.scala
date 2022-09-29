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

class GpuGenerateSuite
  extends SparkQueryCompareTestSuite
    with Arm {
  val rapidsConf = new RapidsConf(Map.empty[String, String])

  def makeListColumn(numRows: Int, listSize: Int): ColumnVector = {
    val list = util.Arrays.asList((0 until listSize):_*)
    val rows = (0 until numRows).map(_ => list)
    ColumnVector.fromLists(
      new ListType(true,
        new BasicType(true, DType.INT32)),
      rows: _*)
  }

  def makeBatch(numRows: Int,
                includeRepeatColumn: Boolean,
                includeNulls: Boolean = false,
                listSize: Int = 4): (Long, ColumnarBatch) = {
    var inputSize: Long = 0L
    withResource(makeListColumn(numRows, listSize)) { cvList =>
      inputSize +=
        withResource(cvList.getChildColumnView(0)) {
          _.getDeviceMemorySize
        }
      val batch = if (includeRepeatColumn) {
        val dt: Array[DataType] = Seq(IntegerType, ArrayType(IntegerType)).toArray
        val secondColumn = (0 until numRows).map { x =>
          val i = Int.box(x)
          if (includeNulls) {
            if (i % 2 == 0) {
              null
            } else {
              i
            }
          } else {
            i
          }
        }
        withResource(ColumnVector.fromBoxedInts(secondColumn: _*)) { repeatCol =>
          inputSize += listSize * repeatCol.getDeviceMemorySize
          GpuColumnVector.from(new Table(repeatCol, cvList), dt)
        }
      } else {
        val dt: Array[DataType] = Seq(ArrayType(IntegerType)).toArray
        GpuColumnVector.from(new Table(cvList), dt)
      }
      (inputSize, batch)
    }
  }

  def checkSplits(inputSize: Long, inputRowSize: Int,
                  batchSize: Int, splits: Array[Int], batch: ColumnarBatch): Unit = {
    withResource(GpuColumnVector.from(batch)) { tbl =>
      var totalRows = 0L
      val splitted = tbl.contiguousSplit(splits:_*)
      splitted.foreach { ct =>
        withResource(ct) { _ =>
          totalRows += ct.getRowCount
          withResource(ct.getTable) { splitTable =>
            println(s"batchSize: $batchSize, splitSize = ${splitTable.getDeviceMemorySize}")
          }
        }
      }

      assertResult(totalRows)(batch.numRows())
    }
  }

  test("0-row batches fallback to no splits") {
    val (inputSize, batch) = makeBatch(numRows = 0, includeRepeatColumn = true)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      assertResult(0)(
        e.inputSplitIndices(batch, 1, false, 1).length)
      assertResult(0)(
        e.inputSplitIndices(batch, generatorOffset = 1 , true, 1).length)
    }
  }

  test("1-row batches fallback to no splits") {
    val (inputSize, batch) = makeBatch(numRows = 1, includeRepeatColumn = true)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      assertResult(0)(
        e.inputSplitIndices(batch, 1, false, 1).length)
      assertResult(0)(
        e.inputSplitIndices(batch, generatorOffset = 1 , true, 1).length)
    }
  }

  test("test batch with a single exploding column") {
    val (inputSize, batch) = makeBatch(numRows=100, includeRepeatColumn=false)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      // the exploded column should be 4 Bytes * 100 rows * 4 reps per row = 1600 Bytes.
      // 1600 == a single split
      assertResult(0)(
        e.inputSplitIndices(batch, 0, false, 1600).length)

      // 800 == 1 splits (2 parts) right down the middle
      var splits = e.inputSplitIndices(batch, 0, false, 800)
      println(splits)
      assertResult(1)(splits.length)
      assertResult(50)(splits(0))

      // 400 == 3 splits (4 parts)
      splits = e.inputSplitIndices(batch, 0, false, 400)
      assertResult(3)(splits.length)
      assertResult(25)(splits(0))
      assertResult(50)(splits(1))
      assertResult(75)(splits(2))
    }
  }

  test("test batch with a two a repeating column") {
    val (inputSize, batch) = makeBatch(numRows=100, includeRepeatColumn=true)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      // the exploded column should be 4 Bytes * 100 rows * 4 reps per row = 1600 Bytes.
      // the repeating column is 4 bytes (or 400 bytes total) repeated 4 times (or 1600)
      // 1600 == two splits
      var splits = e.inputSplitIndices(batch, 1, false, 1600)
      checkSplits(inputSize, 32, 1600, splits, batch)

      // 1600 == 1 splits (2 parts) right down the middle
      splits = e.inputSplitIndices(batch, 1, false, 800)
      checkSplits(3200, 32, 800, splits, batch)

      // 800 == 3 splits (4 parts)
      splits = e.inputSplitIndices(batch, 1, false, 400)
      checkSplits(3200, 32, 400, splits, batch)

      splits = e.inputSplitIndices(batch, 1, false, 100)
      checkSplits(3200, 32, 100, splits, batch)
    }
  }

  test("test batch with a two a repeating column with nulls") {
    val (inputSize, batch) = makeBatch(numRows=100, includeRepeatColumn = true, includeNulls = true)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      // the exploded column should be 4 Bytes * 100 rows * 4 reps per row = 1600 Bytes.
      // the repeating column is 4 bytes (or 400 bytes total) repeated 4 times (or 1600)
      // 1600 == two splits
      var splits = e.inputSplitIndices(batch, 1, false, 1600)
      checkSplits(inputSize, 32, 1600, splits, batch)

      // 1600 == 1 splits (2 parts) right down the middle
      splits = e.inputSplitIndices(batch, 1, false, 800)
      checkSplits(inputSize, 32, 800, splits, batch)

      // 800 == 3 splits (4 parts)
      splits = e.inputSplitIndices(batch, 1, false, 400)
      checkSplits(inputSize, 32, 400, splits, batch)

      splits = e.inputSplitIndices(batch, 1, false, 100)
      checkSplits(inputSize, 32, 100, splits, batch)
    }
  }

  test("test batch with a two a repeating column with nulls and outer") {
    val (inputSize, batch) =
      makeBatch(numRows = 100, includeRepeatColumn = true, includeNulls = true)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      // the exploded column should be 4 Bytes * 100 rows * 4 reps per row = 1600 Bytes.
      // the repeating column is 4 bytes (or 400 bytes total) repeated 4 times (or 1600)
      // 1600 == two splits
      var splits = e.inputSplitIndices(batch, 1, true, 1600)
      checkSplits(inputSize, 32, 1600, splits, batch)

      // 1600 == 1 splits (2 parts) right down the middle
      splits = e.inputSplitIndices(batch, 1, true, 800)
      checkSplits(inputSize, 32, 800, splits, batch)

      // 800 == 3 splits (4 parts)
      splits = e.inputSplitIndices(batch, 1, true, 400)
      checkSplits(inputSize, 32, 400, splits, batch)

      splits = e.inputSplitIndices(batch, 1, true, 100)
      checkSplits(inputSize, 32, 100, splits, batch)
    }
  }

  test("test 1000 row batch with a two a repeating column with nulls and outer") {
    val (inputSize, batch) =
      makeBatch(numRows = 10000, includeRepeatColumn = true, includeNulls = true)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      // the exploded column should be 4 Bytes * 100 rows * 4 reps per row = 1600 Bytes.
      // the repeating column is 4 bytes (or 400 bytes total) repeated 4 times (or 1600)
      // 1600 == two splits
      var splits = e.inputSplitIndices(batch, 1, true, 1600)
      checkSplits(inputSize, 32, 1600, splits, batch)

      // 1600 == 1 splits (2 parts) right down the middle
      splits = e.inputSplitIndices(batch, 1, true, 800)
      checkSplits(inputSize, 32, 800, splits, batch)

      // 800 == 3 splits (4 parts)
      splits = e.inputSplitIndices(batch, 1, true, 400)
      checkSplits(inputSize, 32, 400, splits, batch)

      splits = e.inputSplitIndices(batch, 1, true, 100)
      checkSplits(inputSize, 32, 100, splits, batch)
    }
  }
}
