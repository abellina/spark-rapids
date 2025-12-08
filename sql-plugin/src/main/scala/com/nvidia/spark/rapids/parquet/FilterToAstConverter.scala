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

import java.sql.{Date, Timestamp}
import java.util.Locale

import ai.rapids.cudf.DType
import ai.rapids.cudf.ast.{AstExpression, BinaryOperation, BinaryOperator, ColumnReference, CompiledExpression, Literal, UnaryOperation, UnaryOperator}

import org.apache.spark.internal.Logging
import org.apache.spark.sql.sources._
import org.apache.spark.sql.types._

/**
 * Converts Spark SQL Filter expressions to cuDF AST expressions for use in
 * Parquet hybrid scan filtering.
 *
 * This converter handles the translation of Spark's Filter objects (from the
 * DataSource API v1) into cuDF's AST expression format, which can then be
 * passed to the Parquet reader for statistics-based filtering, dictionary
 * filtering, and optionally bloom filter checking.
 *
 * @param schema The schema of the Parquet file being read, used for column
 *               index lookup and type information
 * @param isCaseSensitive Whether column name matching should be case-sensitive
 */
case class FilterToAstConverter(
    schema: StructType,
    isCaseSensitive: Boolean) extends Logging {

  // Build a map from column name to index for efficient lookups
  private val columnNameToIndex: Map[String, Int] = {
    val fields = schema.fields.zipWithIndex
    if (isCaseSensitive) {
      fields.map { case (field, idx) => field.name -> idx }.toMap
    } else {
      fields.map { case (field, idx) => field.name.toLowerCase(Locale.ROOT) -> idx }.toMap
    }
  }

  // Build a map from column name to data type
  private val columnNameToType: Map[String, DataType] = {
    val fields = schema.fields
    if (isCaseSensitive) {
      fields.map(f => f.name -> f.dataType).toMap
    } else {
      fields.map(f => f.name.toLowerCase(Locale.ROOT) -> f.dataType).toMap
    }
  }

  /**
   * Convert an array of Spark Filters to a single compiled cuDF AST expression.
   * Multiple filters are combined with AND operations.
   *
   * @param filters The filters to convert
   * @return Some(CompiledExpression) if conversion was successful, None if
   *         any filter could not be converted
   */
  def convert(filters: Array[Filter]): Option[CompiledExpression] = {
    if (filters.isEmpty) {
      return None
    }

    val convertedFilters = filters.flatMap(convertFilter)
    if (convertedFilters.isEmpty) {
      return None
    }

    // Combine all filters with AND
    val combined = convertedFilters.reduce { (left, right) =>
      new BinaryOperation(BinaryOperator.LOGICAL_AND, left, right)
    }

    Some(combined.compile())
  }

  /**
   * Convert a single Spark Filter to a cuDF AST expression.
   *
   * @param filter The filter to convert
   * @return Some(AstExpression) if conversion was successful, None if the
   *         filter type is not supported
   */
  def convertFilter(filter: Filter): Option[AstExpression] = {
    filter match {
      // Comparison filters
      case EqualTo(attribute, value) =>
        convertComparison(attribute, value, BinaryOperator.EQUAL)

      case EqualNullSafe(attribute, value) =>
        convertComparison(attribute, value, BinaryOperator.NULL_EQUAL)

      case GreaterThan(attribute, value) =>
        convertComparison(attribute, value, BinaryOperator.GREATER)

      case GreaterThanOrEqual(attribute, value) =>
        convertComparison(attribute, value, BinaryOperator.GREATER_EQUAL)

      case LessThan(attribute, value) =>
        convertComparison(attribute, value, BinaryOperator.LESS)

      case LessThanOrEqual(attribute, value) =>
        convertComparison(attribute, value, BinaryOperator.LESS_EQUAL)

      // Null checks
      case IsNull(attribute) =>
        getColumnIndex(attribute).map { idx =>
          new UnaryOperation(UnaryOperator.IS_NULL, new ColumnReference(idx))
        }

      case IsNotNull(attribute) =>
        getColumnIndex(attribute).map { idx =>
          new UnaryOperation(
            UnaryOperator.NOT,
            new UnaryOperation(UnaryOperator.IS_NULL, new ColumnReference(idx)))
        }

      // IN filter - convert to chain of OR(EQUAL, EQUAL, ...)
      case In(attribute, values) =>
        convertIn(attribute, values)

      // Compound filters
      case And(left, right) =>
        for {
          leftExpr <- convertFilter(left)
          rightExpr <- convertFilter(right)
        } yield new BinaryOperation(BinaryOperator.LOGICAL_AND, leftExpr, rightExpr)

      case Or(left, right) =>
        for {
          leftExpr <- convertFilter(left)
          rightExpr <- convertFilter(right)
        } yield new BinaryOperation(BinaryOperator.LOGICAL_OR, leftExpr, rightExpr)

      case Not(child) =>
        convertFilter(child).map { childExpr =>
          new UnaryOperation(UnaryOperator.NOT, childExpr)
        }

      // String predicates are not supported initially
      case StringStartsWith(_, _) =>
        logDebug(s"StringStartsWith filter not supported for hybrid scan: $filter")
        None

      case StringEndsWith(_, _) =>
        logDebug(s"StringEndsWith filter not supported for hybrid scan: $filter")
        None

      case StringContains(_, _) =>
        logDebug(s"StringContains filter not supported for hybrid scan: $filter")
        None

      case other =>
        logDebug(s"Unsupported filter for hybrid scan: $other")
        None
    }
  }

  /**
   * Convert a comparison filter (EqualTo, GreaterThan, etc.) to an AST expression.
   */
  private def convertComparison(
      attribute: String,
      value: Any,
      operator: BinaryOperator): Option[AstExpression] = {
    for {
      colIdx <- getColumnIndex(attribute)
      dataType <- getColumnType(attribute)
      literal <- convertLiteral(value, dataType)
    } yield {
      new BinaryOperation(operator, new ColumnReference(colIdx), literal)
    }
  }

  /**
   * Convert an IN filter to a chain of OR(EQUAL, EQUAL, ...) expressions.
   */
  private def convertIn(attribute: String, values: Array[Any]): Option[AstExpression] = {
    if (values.isEmpty) {
      return None
    }

    val colIdxOpt = getColumnIndex(attribute)
    val dataTypeOpt = getColumnType(attribute)

    (colIdxOpt, dataTypeOpt) match {
      case (Some(colIdx), Some(dataType)) =>
        // Convert each value to an EQUAL comparison
        val equalComparisons = values.flatMap { value =>
          convertLiteral(value, dataType).map { literal =>
            new BinaryOperation(
              BinaryOperator.EQUAL,
              new ColumnReference(colIdx),
              literal): AstExpression
          }
        }

        if (equalComparisons.isEmpty) {
          None
        } else if (equalComparisons.length == 1) {
          Some(equalComparisons.head)
        } else {
          // Combine with OR
          Some(equalComparisons.reduce { (left, right) =>
            new BinaryOperation(BinaryOperator.LOGICAL_OR, left, right)
          })
        }

      case _ => None
    }
  }

  /**
   * Get the column index for an attribute name.
   */
  private def getColumnIndex(attribute: String): Option[Int] = {
    val key = if (isCaseSensitive) attribute else attribute.toLowerCase(Locale.ROOT)
    columnNameToIndex.get(key)
  }

  /**
   * Get the data type for an attribute name.
   */
  private def getColumnType(attribute: String): Option[DataType] = {
    val key = if (isCaseSensitive) attribute else attribute.toLowerCase(Locale.ROOT)
    columnNameToType.get(key)
  }

  /**
   * Convert a Spark value to a cuDF AST Literal based on the column's data type.
   */
  private def convertLiteral(value: Any, dataType: DataType): Option[Literal] = {
    if (value == null) {
      return Some(Literal.ofNull(sparkTypeToCudfType(dataType)))
    }

    try {
      dataType match {
        case BooleanType =>
          Some(Literal.ofBoolean(value.asInstanceOf[Boolean]))

        case ByteType =>
          Some(Literal.ofByte(value.asInstanceOf[Number].byteValue()))

        case ShortType =>
          Some(Literal.ofShort(value.asInstanceOf[Number].shortValue()))

        case IntegerType =>
          Some(Literal.ofInt(value.asInstanceOf[Number].intValue()))

        case LongType =>
          Some(Literal.ofLong(value.asInstanceOf[Number].longValue()))

        case FloatType =>
          Some(Literal.ofFloat(value.asInstanceOf[Number].floatValue()))

        case DoubleType =>
          Some(Literal.ofDouble(value.asInstanceOf[Number].doubleValue()))

        case StringType =>
          Some(Literal.ofString(value.toString))

        case DateType =>
          // DateType in Spark is stored as days since epoch
          value match {
            case d: Date =>
              Some(Literal.ofTimestampDaysFromInt((d.getTime / (24 * 60 * 60 * 1000)).toInt))
            case i: Number =>
              Some(Literal.ofTimestampDaysFromInt(i.intValue()))
            case _ =>
              logDebug(s"Unsupported date value type: ${value.getClass}")
              None
          }

        case TimestampType =>
          // TimestampType in Spark is stored as microseconds since epoch
          value match {
            case ts: Timestamp =>
              Some(Literal.ofTimestampFromLong(
                DType.TIMESTAMP_MICROSECONDS, ts.getTime * 1000 + (ts.getNanos / 1000) % 1000))
            case l: Number =>
              Some(Literal.ofTimestampFromLong(DType.TIMESTAMP_MICROSECONDS, l.longValue()))
            case _ =>
              logDebug(s"Unsupported timestamp value type: ${value.getClass}")
              None
          }

        case _: DecimalType =>
          // Decimal support may need special handling based on precision/scale
          logDebug(s"DecimalType not yet fully supported in hybrid scan filter: $value")
          None

        case other =>
          logDebug(s"Unsupported data type for hybrid scan filter: $other")
          None
      }
    } catch {
      case e: Exception =>
        logDebug(s"Failed to convert literal value $value to type $dataType: ${e.getMessage}")
        None
    }
  }

  /**
   * Convert a Spark DataType to a cuDF DType for creating null literals.
   */
  private def sparkTypeToCudfType(dataType: DataType): DType = {
    dataType match {
      case BooleanType => DType.BOOL8
      case ByteType => DType.INT8
      case ShortType => DType.INT16
      case IntegerType => DType.INT32
      case LongType => DType.INT64
      case FloatType => DType.FLOAT32
      case DoubleType => DType.FLOAT64
      case StringType => DType.STRING
      case DateType => DType.TIMESTAMP_DAYS
      case TimestampType => DType.TIMESTAMP_MICROSECONDS
      case _ => DType.STRING // Fallback
    }
  }
}

/**
 * Companion object with utility methods for filter conversion.
 */
object FilterToAstConverter {

  /**
   * Check if a filter can be potentially converted to an AST expression.
   * This is a quick check that doesn't actually perform the conversion.
   */
  def isSupportedFilter(filter: Filter): Boolean = {
    filter match {
      case EqualTo(_, _) => true
      case EqualNullSafe(_, _) => true
      case GreaterThan(_, _) => true
      case GreaterThanOrEqual(_, _) => true
      case LessThan(_, _) => true
      case LessThanOrEqual(_, _) => true
      case IsNull(_) => true
      case IsNotNull(_) => true
      case In(_, _) => true
      case And(left, right) => isSupportedFilter(left) && isSupportedFilter(right)
      case Or(left, right) => isSupportedFilter(left) && isSupportedFilter(right)
      case Not(child) => isSupportedFilter(child)
      case _ => false
    }
  }

  /**
   * Get the set of column names referenced by a filter.
   */
  def getReferencedColumns(filter: Filter): Set[String] = {
    filter match {
      case EqualTo(attribute, _) => Set(attribute)
      case EqualNullSafe(attribute, _) => Set(attribute)
      case GreaterThan(attribute, _) => Set(attribute)
      case GreaterThanOrEqual(attribute, _) => Set(attribute)
      case LessThan(attribute, _) => Set(attribute)
      case LessThanOrEqual(attribute, _) => Set(attribute)
      case IsNull(attribute) => Set(attribute)
      case IsNotNull(attribute) => Set(attribute)
      case In(attribute, _) => Set(attribute)
      case StringStartsWith(attribute, _) => Set(attribute)
      case StringEndsWith(attribute, _) => Set(attribute)
      case StringContains(attribute, _) => Set(attribute)
      case And(left, right) => getReferencedColumns(left) ++ getReferencedColumns(right)
      case Or(left, right) => getReferencedColumns(left) ++ getReferencedColumns(right)
      case Not(child) => getReferencedColumns(child)
      case _ => Set.empty
    }
  }
}

