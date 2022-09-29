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

import ai.rapids.cudf.{ColumnVector, ContiguousTable, DType, Table}
import ai.rapids.cudf.HostColumnVector.{BasicType, ListType}

import java.util
import org.apache.spark.sql.types.{ArrayType, DataType, IntegerType}
import org.apache.spark.sql.vectorized.ColumnarBatch

class GpuGenerateSuite
  extends SparkQueryCompareTestSuite
    with Arm {
  val rapidsConf = new RapidsConf(Map.empty[String, String])

  def makeListColumn(numRows: Int, listSize: Int, includeNulls: Boolean): ColumnVector = {
    val list = util.Arrays.asList((0 until listSize):_*)
    val rows = (0 until numRows).map { r =>
      if (includeNulls && r % 2 == 0) {
        null
      } else {
        list
      }
    }
    ColumnVector.fromLists(
      new ListType(true,
        new BasicType(true, DType.INT32)),
      rows: _*)
  }

  def makeBatch(numRows: Int,
                includeRepeatColumn: Boolean = true,
                includeNulls: Boolean = false,
                listSize: Int = 4): (Long, ColumnarBatch) = {
    var inputSize: Long = 0L
    withResource(makeListColumn(numRows, listSize, includeNulls)) { cvList =>
      inputSize +=
        withResource(cvList.getChildColumnView(0)) {
          _.getDeviceMemorySize
        }
      val batch = if (includeRepeatColumn) {
        val dt: Array[DataType] = Seq(IntegerType, ArrayType(IntegerType)).toArray
        val secondColumn = (0 until numRows).map { x =>
          val i = Int.box(x)
          if (includeNulls && i % 2 == 0) {
            null
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

  def checkSplits(splits: Array[Int], batch: ColumnarBatch): Unit = {
    withResource(GpuColumnVector.from(batch)) { tbl =>
      var totalRows = 0L
      val splitted: Array[ContiguousTable] = tbl.contiguousSplit(splits: _*)
      splitted.foreach { ct =>
        totalRows += ct.getRowCount
      }
      withResource(splitted) { _ =>
        // `getTable` causes Table to be owned by the `ContiguousTable` class
        // so they get closed when the `ContiguousTable`s get closed.
        val concatted = if (splitted.length == 1) {
          splitted(0).getTable
        } else {
          Table.concatenate(splitted.map(_.getTable): _*)
        }

        // Compare row by row the input vs the concatenated splits
        withResource(GpuColumnVector.from(batch)) { inputTbl =>
          assertResult(concatted.getRowCount)(batch.numRows())
          (0 until batch.numCols()).foreach { c =>
            withResource(concatted.getColumn(c).copyToHost()) { hostConcatCol =>
              withResource(inputTbl.getColumn(c).copyToHost()) { hostInputCol =>
                (0 until batch.numRows()).foreach { r =>
                  if (hostInputCol.isNull(r)) {
                    assertResult(true)(hostConcatCol.isNull(r))
                  } else {
                    if (hostInputCol.getType == DType.LIST) {
                      // exploding column
                      compare(hostInputCol.getList(r), hostConcatCol.getList(r))
                    } else {
                      compare(hostInputCol.getInt(r), hostConcatCol.getInt(r))
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // test a short-circuit path
  test("0-row batches short-circuits to no splits") {
    val (inputSize, batch) = makeBatch(numRows = 0)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      assertResult(0)(
        e.inputSplitIndices(batch, 1, false, 1).length)
      assertResult(0)(
        e.inputSplitIndices(batch, generatorOffset = 1 , true, 1).length)
    }
  }

  // test a short-circuit path
  test("1-row batches short-circuits to no splits") {
    val (inputSize, batch) = makeBatch(numRows = 1)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      val target = inputSize / 2
      var splits = e.inputSplitIndices(batch, 1, false, target)
      assertResult(0)(splits.length)
      checkSplits(splits, batch)

      splits = e.inputSplitIndices(batch, generatorOffset = 1 , true, target)
      assertResult(0)(splits.length)
      checkSplits(splits, batch)
    }
  }

  test("2-row batches split in half") {
    val (inputSize, batch) = makeBatch(numRows = 2)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      val target = inputSize/2
      var splits = e.inputSplitIndices(batch, 1, false, target)
      assertResult(1)(splits.length)
      assertResult(1)(splits(0))
      checkSplits(splits, batch)

      splits = e.inputSplitIndices(batch, generatorOffset = 1 , true, target)
      assertResult(1)(splits.length)
      assertResult(1)(splits(0))
      checkSplits(splits, batch)
    }
  }

  test("4-row batches split in half") {
    val (inputSize, batch) = makeBatch(numRows = 8)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      val target = inputSize/2
      var splits = e.inputSplitIndices(batch, 1, false, target)
      assertResult(1)(splits.length)
      assertResult(4)(splits(0))
      checkSplits(splits, batch)

      splits = e.inputSplitIndices(batch, generatorOffset = 1 , true, target)
      assertResult(1)(splits.length)
      assertResult(4)(splits(0))
      checkSplits(splits, batch)
    }
  }

  // this tests part of the code that just uses the exploding column's size as the limit
  test("test batch with a single exploding column") {
    val (inputSize, batch) = makeBatch(numRows=100, includeRepeatColumn=false)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      // the exploded column should be 4 Bytes * 100 rows * 4 reps per row = 1600 Bytes.
      // 1600 == a single split
      var splits = e.inputSplitIndices(batch, 0, false, 1600)
      assertResult(0)(splits.length)
      checkSplits(splits, batch)

      // 800 == 1 splits (2 parts) right down the middle
      splits = e.inputSplitIndices(batch, 0, false, 800)
      println(splits)
      assertResult(1)(splits.length)
      assertResult(50)(splits(0))
      checkSplits(splits, batch)

      // 400 == 3 splits (4 parts)
      splits = e.inputSplitIndices(batch, 0, false, 400)
      assertResult(3)(splits.length)
      assertResult(25)(splits(0))
      assertResult(50)(splits(1))
      assertResult(75)(splits(2))
      checkSplits(splits, batch)
    }
  }

  test("test batch with a single exploding column with nulls") {
    val (inputSize, batch) = makeBatch(numRows=100, includeRepeatColumn=false, includeNulls=true)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      // the exploded column should be 4 Bytes * 100 rows * 4 reps per row = 1600 Bytes.
      // 1600 == a single split
      var splits = e.inputSplitIndices(batch, 0, false, 800, maxRows = 50)
      assertResult(3)(splits.length)
      assertResult(25)(splits(0))
      assertResult(50)(splits(1))
      assertResult(75)(splits(2))
      checkSplits(splits, batch)

      // 800 == 1 splits (2 parts) right down the middle
      splits = e.inputSplitIndices(batch, 0, false, 400, maxRows = 50)
      assertResult(3)(splits.length)
      assertResult(25)(splits(0))
      assertResult(50)(splits(1))
      assertResult(75)(splits(2))
      checkSplits(splits, batch)

      // 100 == 8 parts
      splits = e.inputSplitIndices(batch, 0, false, 100, maxRows = 50)
      assertResult(7)(splits.length)
      assertResult(13)(splits(0))
      assertResult(25)(splits(1))
      assertResult(38)(splits(2))
      assertResult(50)(splits(3))
      assertResult(63)(splits(4))
      assertResult(75)(splits(5))
      assertResult(88)(splits(6))
      checkSplits(splits, batch)
    }
  }

  test("outer: test batch with a single exploding column with nulls") {
    val (inputSize, batch) = makeBatch(numRows=100, includeRepeatColumn=false, includeNulls=true)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      // the exploded column should be 4 Bytes * 100 rows * 4 reps per row = 1600 Bytes.
      // 1600 == a single split
      assertResult(0)(
        e.inputSplitIndices(batch, 0, true, 800, maxRows = 50).length)

      // 800 == 1 splits (2 parts) right down the middle
      var splits = e.inputSplitIndices(batch, 0, true, 400, maxRows = 50)
      assertResult(1)(splits.length)
      assertResult(50)(splits(0))

      // 400 == 3 splits (4 parts)
      splits = e.inputSplitIndices(batch, 0, true, 200, maxRows = 50)
      assertResult(3)(splits.length)
      assertResult(25)(splits(0))
      assertResult(50)(splits(1))
      assertResult(75)(splits(2))
    }
  }

  test("test batch with a two a repeating column") {
    val (inputSize, batch) = makeBatch(numRows=100)
    withResource(batch) { _ =>
      val e = GpuExplode(null)

      // no splits
      var splits = e.inputSplitIndices(batch, 1, false, inputSize)
      assertResult(0)(splits.length)
      checkSplits(splits, batch)

      // 1600 == 1 splits (2 parts) right down the middle
      var numParts = 2
      splits = e.inputSplitIndices(batch, 1, false, inputSize/numParts)
      assertResult(1)(splits.length)
      assertResult(50)(splits(0))
      checkSplits(splits, batch)

      numParts = 4
      splits = e.inputSplitIndices(batch, 1, false, inputSize/numParts)
      assertResult(3)(splits.length)
      assertResult(25)(splits(0))
      assertResult(50)(splits(1))
      assertResult(75)(splits(2))
      checkSplits(splits, batch)

      numParts = 8
      splits = e.inputSplitIndices(batch, 1, false, inputSize/numParts)
      assertResult(7)(splits.length)
      assertResult(13)(splits(0))
      assertResult(25)(splits(1))
      assertResult(38)(splits(2))
      assertResult(50)(splits(3))
      assertResult(63)(splits(4))
      assertResult(75)(splits(5))
      assertResult(88)(splits(6))
      checkSplits(splits, batch)
    }
  }

  test("test batch with a two a repeating column with nulls") {
    val (inputSize, batch) = makeBatch(numRows=100, includeNulls = true)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      // the exploded column should be 4 Bytes * 100 rows * 4 reps per row = 1600 Bytes.
      // the repeating column is 4 bytes (or 400 bytes total) repeated 4 times (or 1600)
      // 1600 == two splits
      var splits = e.inputSplitIndices(batch, 1, false, 1600)
      checkSplits(splits, batch)

      // 1600 == 1 splits (2 parts) right down the middle
      splits = e.inputSplitIndices(batch, 1, false, 800)
      checkSplits(splits, batch)

      // 800 == 3 splits (4 parts)
      splits = e.inputSplitIndices(batch, 1, false, 400)
      checkSplits(splits, batch)

      // split at every row (rows - 1)
      splits = e.inputSplitIndices(batch, 1, true, 1)
      assertResult(splits.length)(50)
      checkSplits(splits, batch)

      splits = e.inputSplitIndices(batch, 1, false, 1)
      assertResult(splits.length)(99)
      checkSplits(splits, batch)
    }
  }

  test("test batch with a two a repeating column with nulls and outer") {
    val (inputSize, batch) =
      makeBatch(numRows = 100, includeNulls = true)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      // the exploded column should be 4 Bytes * 100 rows * 4 reps per row = 1600 Bytes.
      // the repeating column is 4 bytes (or 400 bytes total) repeated 4 times (or 1600)
      // 1600 == two splits
      var splits = e.inputSplitIndices(batch, 1, true, 1600)
      checkSplits(splits, batch)

      // 1600 == 1 splits (2 parts) right down the middle
      splits = e.inputSplitIndices(batch, 1, true, 800)
      checkSplits(splits, batch)

      // 800 == 3 splits (4 parts)
      splits = e.inputSplitIndices(batch, 1, true, 400)
      checkSplits(splits, batch)

      splits = e.inputSplitIndices(batch, 1, true, 100)
      checkSplits(splits, batch)
    }
  }

  test("test 1000 row batch with a two a repeating column with nulls and outer") {
    val (inputSize, batch) =
      makeBatch(numRows = 10000, includeNulls = true)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      // the exploded column should be 4 Bytes * 100 rows * 4 reps per row = 1600 Bytes.
      // the repeating column is 4 bytes (or 400 bytes total) repeated 4 times (or 1600)
      // 1600 == two splits
      var splits = e.inputSplitIndices(batch, 1, true, 1600)
      checkSplits(splits, batch)

      // 1600 == 1 splits (2 parts) right down the middle
      splits = e.inputSplitIndices(batch, 1, true, 800)
      checkSplits(splits, batch)

      // 800 == 3 splits (4 parts)
      splits = e.inputSplitIndices(batch, 1, true, 400)
      checkSplits(splits, batch)

      splits = e.inputSplitIndices(batch, 1, true, 100)
      checkSplits(splits, batch)
    }
  }

  test("if the row limit produces more splits, prefer splitting using maxRows") {
    val (inputSize, batch) =
      makeBatch(numRows = 10000, includeNulls = true)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      // by size try to no splits, instead we should get 2 (by maxRows)
      // we expect 40000 rows (4x1000 given 4 items in the list per row), but we included nulls,
      // so this should return 20000 rows (given that this is not outer)
      var splits = e.inputSplitIndices(batch, 1, false, inputSize, maxRows = 20000)
      assertResult(0)(splits.length)
      checkSplits(splits, batch)

      splits = e.inputSplitIndices(batch, 1, false, inputSize, maxRows = 10000)
      assertResult(1)(splits.length)
      assertResult(5000)(splits(0))
      checkSplits(splits, batch)
    }
  }

  test("outer: if the row limit produces more splits, prefer splitting using maxRows") {
    val (inputSize, batch) =
      makeBatch(numRows = 10000, includeNulls = true)
    withResource(batch) { _ =>
      val e = GpuExplode(null)
      // by size try to no splits, instead we should get 2 (by maxRows)
      // we expect 40000 rows (4x1000 given 4 items in the list per row)
      val splits = e.inputSplitIndices(batch, 1, true, inputSize, maxRows = 20000)
      assertResult(1)(splits.length)
      assertResult(5000)(splits(0))
      checkSplits(splits, batch)
    }
  }
}
