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

package com.nvidia.spark.rapids.parquet

import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.sources._
import org.apache.spark.sql.types._

class FilterToAstConverterSuite extends AnyFunSuite {

  private val testSchema = StructType(Seq(
    StructField("int_col", IntegerType),
    StructField("long_col", LongType),
    StructField("float_col", FloatType),
    StructField("double_col", DoubleType),
    StructField("string_col", StringType),
    StructField("bool_col", BooleanType),
    StructField("date_col", DateType),
    StructField("ts_col", TimestampType)
  ))

  private def converter(caseSensitive: Boolean = true, useColumnNames: Boolean = false) = 
    FilterToAstConverter(testSchema, caseSensitive, useColumnNames)

  // Test supported filter types
  test("isSupportedFilter returns true for comparison filters") {
    assert(FilterToAstConverter.isSupportedFilter(EqualTo("int_col", 1)))
    assert(FilterToAstConverter.isSupportedFilter(EqualNullSafe("int_col", 1)))
    assert(FilterToAstConverter.isSupportedFilter(GreaterThan("int_col", 1)))
    assert(FilterToAstConverter.isSupportedFilter(GreaterThanOrEqual("int_col", 1)))
    assert(FilterToAstConverter.isSupportedFilter(LessThan("int_col", 1)))
    assert(FilterToAstConverter.isSupportedFilter(LessThanOrEqual("int_col", 1)))
  }

  test("isSupportedFilter returns true for null checks") {
    assert(FilterToAstConverter.isSupportedFilter(IsNull("int_col")))
    assert(FilterToAstConverter.isSupportedFilter(IsNotNull("int_col")))
  }

  test("isSupportedFilter returns true for In filter") {
    assert(FilterToAstConverter.isSupportedFilter(In("int_col", Array(1, 2, 3))))
  }

  test("isSupportedFilter returns true for compound filters") {
    assert(FilterToAstConverter.isSupportedFilter(
      And(EqualTo("int_col", 1), GreaterThan("long_col", 100L))))
    assert(FilterToAstConverter.isSupportedFilter(
      Or(EqualTo("int_col", 1), EqualTo("int_col", 2))))
    assert(FilterToAstConverter.isSupportedFilter(
      Not(EqualTo("int_col", 1))))
  }

  test("isSupportedFilter returns false for string predicates") {
    assert(!FilterToAstConverter.isSupportedFilter(StringStartsWith("string_col", "foo")))
    assert(!FilterToAstConverter.isSupportedFilter(StringEndsWith("string_col", "bar")))
    assert(!FilterToAstConverter.isSupportedFilter(StringContains("string_col", "baz")))
  }

  // Test conversion to AST
  test("convertFilter returns Some for EqualTo with integer") {
    val result = converter().convertFilter(EqualTo("int_col", 42))
    assert(result.isDefined)
  }

  test("convertFilter returns Some for EqualTo with long") {
    val result = converter().convertFilter(EqualTo("long_col", 100L))
    assert(result.isDefined)
  }

  test("convertFilter returns Some for EqualTo with float") {
    val result = converter().convertFilter(EqualTo("float_col", 3.14f))
    assert(result.isDefined)
  }

  test("convertFilter returns Some for EqualTo with double") {
    val result = converter().convertFilter(EqualTo("double_col", 2.718))
    assert(result.isDefined)
  }

  test("convertFilter returns Some for EqualTo with string") {
    val result = converter().convertFilter(EqualTo("string_col", "test"))
    assert(result.isDefined)
  }

  test("convertFilter returns Some for EqualTo with boolean") {
    val result = converter().convertFilter(EqualTo("bool_col", true))
    assert(result.isDefined)
  }

  test("convertFilter returns Some for comparison operators") {
    assert(converter().convertFilter(GreaterThan("int_col", 10)).isDefined)
    assert(converter().convertFilter(GreaterThanOrEqual("int_col", 10)).isDefined)
    assert(converter().convertFilter(LessThan("int_col", 10)).isDefined)
    assert(converter().convertFilter(LessThanOrEqual("int_col", 10)).isDefined)
  }

  test("convertFilter returns Some for null checks") {
    assert(converter().convertFilter(IsNull("int_col")).isDefined)
    assert(converter().convertFilter(IsNotNull("int_col")).isDefined)
  }

  test("convertFilter returns Some for In filter") {
    val result = converter().convertFilter(In("int_col", Array(1, 2, 3)))
    assert(result.isDefined)
  }

  test("convertFilter returns Some for And filter") {
    val result = converter().convertFilter(
      And(EqualTo("int_col", 1), EqualTo("long_col", 100L)))
    assert(result.isDefined)
  }

  test("convertFilter returns Some for Or filter") {
    val result = converter().convertFilter(
      Or(EqualTo("int_col", 1), EqualTo("int_col", 2)))
    assert(result.isDefined)
  }

  test("convertFilter returns Some for Not filter") {
    val result = converter().convertFilter(Not(EqualTo("int_col", 1)))
    assert(result.isDefined)
  }

  test("convertFilter returns None for unknown column") {
    val result = converter().convertFilter(EqualTo("unknown_col", 1))
    assert(result.isEmpty)
  }

  test("convertFilter handles case insensitivity") {
    val caseSensitiveConverter = converter(caseSensitive = true)
    val caseInsensitiveConverter = converter(caseSensitive = false)
    
    // Case sensitive should not find the column
    assert(caseSensitiveConverter.convertFilter(EqualTo("INT_COL", 1)).isEmpty)
    
    // Case insensitive should find the column
    assert(caseInsensitiveConverter.convertFilter(EqualTo("INT_COL", 1)).isDefined)
  }

  // Test convert method (combines multiple filters)
  test("convert returns None for empty filters") {
    val result = converter().convert(Array.empty)
    assert(result.isEmpty)
  }

  test("convert returns CompiledExpression for single filter") {
    val result = converter().convert(Array(EqualTo("int_col", 42)))
    assert(result.isDefined)
    result.foreach(_.close())
  }

  test("convert returns CompiledExpression for multiple filters") {
    val result = converter().convert(Array(
      EqualTo("int_col", 42),
      GreaterThan("long_col", 100L)
    ))
    assert(result.isDefined)
    result.foreach(_.close())
  }

  test("convert skips unsupported filters") {
    // Mix of supported and unsupported filters
    val result = converter().convert(Array(
      EqualTo("int_col", 42),
      StringStartsWith("string_col", "foo") // unsupported
    ))
    // Should still return a result with the supported filter
    assert(result.isDefined)
    result.foreach(_.close())
  }

  test("convert returns None when all filters are unsupported") {
    val result = converter().convert(Array(
      StringStartsWith("string_col", "foo"),
      StringEndsWith("string_col", "bar")
    ))
    assert(result.isEmpty)
  }

  // Test getReferencedColumns
  test("getReferencedColumns returns correct columns") {
    assert(FilterToAstConverter.getReferencedColumns(EqualTo("int_col", 1)) == Set("int_col"))
    assert(FilterToAstConverter.getReferencedColumns(
      And(EqualTo("int_col", 1), GreaterThan("long_col", 100L))) == Set("int_col", "long_col"))
    assert(FilterToAstConverter.getReferencedColumns(
      Or(EqualTo("int_col", 1), EqualTo("long_col", 100L))) == Set("int_col", "long_col"))
    assert(FilterToAstConverter.getReferencedColumns(
      Not(EqualTo("int_col", 1))) == Set("int_col"))
    assert(FilterToAstConverter.getReferencedColumns(
      In("int_col", Array(1, 2, 3))) == Set("int_col"))
  }
}

