/*
 * Copyright (c) 2022, NVIDIA CORPORATION.
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

package org.apache.spark.sql.types.shims

import java.time.ZoneId

import org.apache.spark.sql.catalyst.catalog.ExternalCatalogUtils.unescapePathName
import org.apache.spark.sql.catalyst.expressions.{Cast, Literal}
import org.apache.spark.sql.types.{AnsiIntervalType, DataType}

object PartitionValueCastShims {
  def isSupportedType(dt: DataType): Boolean = dt match {
    // AnsiIntervalType
    case it: AnsiIntervalType => true
    case _ => false
  }

  // Only for AnsiIntervalType
  def castTo(desiredType: DataType, value: String, zoneId: ZoneId): Any = desiredType match {
    // Copied from org/apache/spark/sql/execution/datasources/PartitionUtils.scala
    case it: AnsiIntervalType =>
      Cast(Literal(unescapePathName(value)), it).eval()
  }
}
