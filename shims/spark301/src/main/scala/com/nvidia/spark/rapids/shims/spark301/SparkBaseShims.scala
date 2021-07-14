/*
 * Copyright (c) 2020-2021, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.shims.spark301

import java.net.URI
import java.nio.ByteBuffer

import com.nvidia.spark.rapids.{GpuSpecialFrameBoundary, _}
import org.apache.arrow.memory.ReferenceManager
import org.apache.arrow.vector.ValueVector
import org.apache.hadoop.fs.Path
import org.apache.spark.SparkEnv

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.{InternalRow, TableIdentifier}
import org.apache.spark.sql.catalyst.analysis.Resolver
import org.apache.spark.sql.catalyst.catalog.{CatalogTable, SessionCatalog}
import org.apache.spark.sql.catalyst.encoders.ExpressionEncoder
import org.apache.spark.sql.catalyst.errors.attachTree
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.plans.JoinType
import org.apache.spark.sql.catalyst.plans.physical.{BroadcastMode, Partitioning}
import org.apache.spark.sql.catalyst.trees.TreeNode
import org.apache.spark.sql.connector.read.Scan
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.adaptive.{BroadcastQueryStageExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.command.{AlterTableRecoverPartitionsCommand, RunnableCommand}
import org.apache.spark.sql.execution.datasources.{FileIndex, FilePartition, FileScanRDD, HadoopFsRelation, InMemoryFileIndex, PartitionDirectory, PartitionedFile}
import org.apache.spark.sql.execution.datasources.rapids.GpuPartitioningUtils
import org.apache.spark.sql.execution.datasources.v2.orc.OrcScan
import org.apache.spark.sql.execution.datasources.v2.parquet.ParquetScan
import org.apache.spark.sql.execution.exchange.{ReusedExchangeExec, ShuffleExchangeExec}
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, BroadcastNestedLoopJoinExec, BuildLeft, BuildRight, BuildSide, HashJoin, ShuffledHashJoinExec, SortMergeJoinExec}
import org.apache.spark.sql.execution.python.{AggregateInPandasExec, ArrowEvalPythonExec, FlatMapGroupsInPandasExec, MapInPandasExec, WindowInPandasExec}
import org.apache.spark.sql.execution.window.WindowExecBase
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.{GpuFileSourceScanExec, GpuStringReplace, GpuTimeSub, ShuffleManagerShimBase}
import org.apache.spark.sql.rapids.execution.{GpuBroadcastExchangeExecBase, GpuBroadcastNestedLoopJoinExec, GpuBroadcastNestedLoopJoinExecBase, GpuShuffleExchangeExecBase}
import org.apache.spark.sql.rapids.execution.python.{GpuAggregateInPandasExecMeta, GpuArrowEvalPythonExec, GpuFlatMapGroupsInPandasExecMeta, GpuMapInPandasExecMeta, GpuPythonUDF, GpuWindowInPandasExecMetaBase}
import org.apache.spark.sql.rapids.shims.spark301.{GpuSchemaUtils, ShuffleManagerShim}
import org.apache.spark.sql.sources.BaseRelation
import org.apache.spark.sql.types._
import org.apache.spark.storage.{BlockId, BlockManagerId}
import org.apache.spark.unsafe.types.CalendarInterval

/**
 * This class contains the default implementation for most shim methods and should be compiled
 * against the lowest supported Spark version. As support for the oldest Spark version is
 * abandoned, this class should move to the next oldest supported version and overrides from
 * that version should be folded into here. Any shim methods that are implemented only in the
 * updated base version can then be removed from the shim interface.
 */
abstract class SparkBaseShims extends SparkCommonShims {


  override def v1RepairTableCommand(tableName: TableIdentifier): RunnableCommand =
    AlterTableRecoverPartitionsCommand(tableName)

  override def getScalaUDFAsExpression(
      function: AnyRef,
      dataType: DataType,
      children: Seq[Expression],
      inputEncoders: Seq[Option[ExpressionEncoder[_]]] = Nil,
      outputEncoder: Option[ExpressionEncoder[_]] = None,
      udfName: Option[String] = None,
      nullable: Boolean = true,
      udfDeterministic: Boolean = true): Expression = {
    // outputEncoder is only used in Spark 3.1+
    ScalaUDF(function, dataType, children, inputEncoders, udfName, nullable, udfDeterministic)
  }

  override def getMapSizesByExecutorId(
      shuffleId: Int,
      startMapIndex: Int,
      endMapIndex: Int,
      startPartition: Int,
      endPartition: Int): Iterator[(BlockManagerId, Seq[(BlockId, Long, Int)])] = {
    SparkEnv.get.mapOutputTracker.getMapSizesByRange(shuffleId,
      startMapIndex, endMapIndex, startPartition, endPartition)
  }

  override def getGpuBroadcastNestedLoopJoinShim(
      left: SparkPlan,
      right: SparkPlan,
      join: BroadcastNestedLoopJoinExec,
      joinType: JoinType,
      condition: Option[Expression],
      targetSizeBytes: Long): GpuBroadcastNestedLoopJoinExecBase = {
    GpuBroadcastNestedLoopJoinExec(left, right, join, joinType, condition, targetSizeBytes)
  }

  override def getGpuShuffleExchangeExec(
      outputPartitioning: Partitioning,
      child: SparkPlan,
      cpuShuffle: Option[ShuffleExchangeExec]): GpuShuffleExchangeExecBase = {
    val canChangeNumPartitions = cpuShuffle.forall(_.canChangeNumPartitions)
    GpuShuffleExchangeExec(outputPartitioning, child, canChangeNumPartitions)
  }

  override def isGpuBroadcastHashJoin(plan: SparkPlan): Boolean = {
    plan match {
      case _: GpuBroadcastHashJoinExec => true
      case _ => false
    }
  }


  override def isGpuShuffledHashJoin(plan: SparkPlan): Boolean = {
    plan match {
      case _: GpuShuffledHashJoinExec => true
      case _ => false
    }
  }

  protected def getExprsSansTimeSub: Map[Class[_ <: Expression], ExprRule[_ <: Expression]] = {
    Seq(
      GpuOverrides.expr[Cast](
        "Convert a column of one type of data into another type",
        new CastChecks(),
        (cast, conf, p, r) => new CastExprMeta[Cast](cast, SparkSession.active.sessionState.conf
            .ansiEnabled, conf, p, r)),
      GpuOverrides.expr[AnsiCast](
        "Convert a column of one type of data into another type",
        new CastChecks(),
        (cast, conf, p, r) => new CastExprMeta[AnsiCast](cast, true, conf, p, r)),
      GpuOverrides.expr[RegExpReplace](
        "RegExpReplace support for string literal input patterns",
        ExprChecks.projectNotLambda(TypeSig.STRING, TypeSig.STRING,
          Seq(ParamCheck("str", TypeSig.STRING, TypeSig.STRING),
            ParamCheck("regex", TypeSig.lit(TypeEnum.STRING)
                .withPsNote(TypeEnum.STRING, "very limited regex support"), TypeSig.STRING),
            ParamCheck("rep", TypeSig.lit(TypeEnum.STRING), TypeSig.STRING))),
        (a, conf, p, r) => new TernaryExprMeta[RegExpReplace](a, conf, p, r) {
          override def tagExprForGpu(): Unit = {
            if (GpuOverrides.isNullOrEmptyOrRegex(a.regexp)) {
              willNotWorkOnGpu(
                "Only non-null, non-empty String literals that are not regex patterns " +
                    "are supported by RegExpReplace on the GPU")
            }
          }
          override def convertToGpu(lhs: Expression, regexp: Expression,
              rep: Expression): GpuExpression = GpuStringReplace(lhs, regexp, rep)
        })
    ).map(r => (r.getClassFor.asSubclass(classOf[Expression]), r)).toMap
  }

  override def getExprs: Map[Class[_ <: Expression], ExprRule[_ <: Expression]] = {
    getExprsSansTimeSub + (classOf[TimeSub] -> GpuOverrides.expr[TimeSub](
      "Subtracts interval from timestamp",
      ExprChecks.binaryProjectNotLambda(TypeSig.TIMESTAMP, TypeSig.TIMESTAMP,
        ("start", TypeSig.TIMESTAMP, TypeSig.TIMESTAMP),
        ("interval", TypeSig.lit(TypeEnum.CALENDAR)
            .withPsNote(TypeEnum.CALENDAR, "months not supported"), TypeSig.CALENDAR)),
      (timeSub, conf, p, r) => new BinaryExprMeta[TimeSub](timeSub, conf, p, r) {
        override def tagExprForGpu(): Unit = {
          timeSub.interval match {
            case Literal(intvl: CalendarInterval, DataTypes.CalendarIntervalType) =>
              if (intvl.months != 0) {
                willNotWorkOnGpu("interval months isn't supported")
              }
            case _ =>
          }
          checkTimeZoneId(timeSub.timeZoneId)
        }

        override def convertToGpu(lhs: Expression, rhs: Expression): GpuExpression =
          GpuTimeSub(lhs, rhs)
      }))
  }

  override def getScans: Map[Class[_ <: Scan], ScanRule[_ <: Scan]] = Seq(
    GpuOverrides.scan[ParquetScan](
      "Parquet parsing",
      (a, conf, p, r) => new ScanMeta[ParquetScan](a, conf, p, r) {
        override def tagSelfForGpu(): Unit = GpuParquetScanBase.tagSupport(this)

        override def convertToGpu(): Scan = {
          GpuParquetScan(a.sparkSession,
            a.hadoopConf,
            a.fileIndex,
            a.dataSchema,
            a.readDataSchema,
            a.readPartitionSchema,
            a.pushedFilters,
            a.options,
            a.partitionFilters,
            a.dataFilters,
            conf)
        }
      }),
    GpuOverrides.scan[OrcScan](
      "ORC parsing",
      (a, conf, p, r) => new ScanMeta[OrcScan](a, conf, p, r) {
        override def tagSelfForGpu(): Unit =
          GpuOrcScanBase.tagSupport(this)

        override def convertToGpu(): Scan =
          GpuOrcScan(a.sparkSession,
            a.hadoopConf,
            a.fileIndex,
            a.dataSchema,
            a.readDataSchema,
            a.readPartitionSchema,
            a.options,
            a.pushedFilters,
            a.partitionFilters,
            a.dataFilters,
            conf)
      })
  ).map(r => (r.getClassFor.asSubclass(classOf[Scan]), r)).toMap

  private def getGpuBuildSide(buildSide: BuildSide): GpuBuildSide = {
    buildSide match {
      case BuildRight => GpuBuildRight
      case BuildLeft => GpuBuildLeft
      case _ => throw new Exception("unknown buildSide Type")
    }
  }

  override def getBuildSide(join: HashJoin): GpuBuildSide = {
    getGpuBuildSide(join.buildSide)
  }

  override def getBuildSide(join: BroadcastNestedLoopJoinExec): GpuBuildSide = {
    getGpuBuildSide(join.buildSide)
  }

  override def getShuffleManagerShims(): ShuffleManagerShimBase = {
    new ShuffleManagerShim
  }


  // Hardcoded for Spark-3.0.*
  override def getFileSourceMaxMetadataValueLength(sqlConf: SQLConf): Int = 100

  override def copyParquetBatchScanExec(
      batchScanExec: GpuBatchScanExec,
      queryUsesInputFile: Boolean): GpuBatchScanExec = {
    val scan = batchScanExec.scan.asInstanceOf[GpuParquetScan]
    val scanCopy = scan.copy(queryUsesInputFile=queryUsesInputFile)
    batchScanExec.copy(scan=scanCopy)
  }

  override def copyFileSourceScanExec(
      scanExec: GpuFileSourceScanExec,
      queryUsesInputFile: Boolean): GpuFileSourceScanExec = {
    scanExec.copy(queryUsesInputFile=queryUsesInputFile)(scanExec.rapidsConf)
  }

  override def getGpuColumnarToRowTransition(plan: SparkPlan,
     exportColumnRdd: Boolean): GpuColumnarToRowExecParent = {
    GpuColumnarToRowExec(plan, exportColumnRdd)
  }

  override def checkColumnNameDuplication(
      schema: StructType,
      colType: String,
      resolver: Resolver): Unit = {
    GpuSchemaUtils.checkColumnNameDuplication(schema, colType, resolver)
  }

  override def sortOrderChildren(s: SortOrder): Seq[Expression] = {
    (s.sameOrderExpressions + s.child).toSeq
  }

  override def sortOrder(
      child: Expression,
      direction: SortDirection,
      nullOrdering: NullOrdering): SortOrder = SortOrder(child, direction, nullOrdering, Set.empty)

  override def copySortOrderWithNewChild(s: SortOrder, child: Expression): SortOrder = {
    s.copy(child = child)
  }

  override def alias(child: Expression, name: String)(
      exprId: ExprId,
      qualifier: Seq[String],
      explicitMetadata: Option[Metadata]): Alias = {
    Alias(child, name)(exprId, qualifier, explicitMetadata)
  }

  override def shouldIgnorePath(path: String): Boolean = {
    InMemoryFileIndex.shouldFilterOut(path)
  }

  override def getLegacyComplexTypeToString(): Boolean = true

  // Arrow version changed between Spark versions
  override def getArrowDataBuf(vec: ValueVector): (ByteBuffer, ReferenceManager) = {
    val arrowBuf = vec.getDataBuffer
    (arrowBuf.nioBuffer(), arrowBuf.getReferenceManager)
  }

  override def getArrowValidityBuf(vec: ValueVector): (ByteBuffer, ReferenceManager) = {
    val arrowBuf = vec.getValidityBuffer
    (arrowBuf.nioBuffer(), arrowBuf.getReferenceManager)
  }

  override def getArrowOffsetsBuf(vec: ValueVector): (ByteBuffer, ReferenceManager) = {
    val arrowBuf = vec.getOffsetBuffer
    (arrowBuf.nioBuffer(), arrowBuf.getReferenceManager)
  }

  override def shouldFailDivByZero(): Boolean = false

  override def reusedExchangeExecPfn: PartialFunction[SparkPlan, ReusedExchangeExec] = {
    case ShuffleQueryStageExec(_, e: ReusedExchangeExec) => e
    case BroadcastQueryStageExec(_, e: ReusedExchangeExec) => e
  }

  /** dropped by SPARK-34234 */
  override def attachTreeIfSupported[TreeType <: TreeNode[_], A](
    tree: TreeType,
    msg: String)(
    f: => A
  ): A = {
    attachTree(tree, msg)(f)
  }

  override def createTable(table: CatalogTable,
    sessionCatalog: SessionCatalog,
    tableLocation: Option[URI],
    result: BaseRelation) = {
    val newTable = table.copy(
      storage = table.storage.copy(locationUri = tableLocation),
      // We will use the schema of resolved.relation as the schema of the table (instead of
      // the schema of df). It is important since the nullability may be changed by the relation
      // provider (for example, see org.apache.spark.sql.parquet.DefaultSource).
      schema = result.schema)
    // Table location is already validated. No need to check it again during table creation.
    sessionCatalog.createTable(newTable, ignoreIfExists = false, validateLocation = false)
  }

  override def boundarySql(expr: Expression) = {
    expr match {
      case e: GpuSpecialFrameBoundary => e.sql
      case UnaryMinus(n) => n.sql + " PRECEDING"
      case e: Expression => e.sql + " FOLLOWING"
    }
  }
}
