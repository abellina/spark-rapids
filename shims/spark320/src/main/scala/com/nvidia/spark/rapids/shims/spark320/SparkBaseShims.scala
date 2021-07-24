/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.shims.spark320

import org.apache.hadoop.fs.Path

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.{InternalRow, TableIdentifier}
import org.apache.spark.sql.catalyst.expressions._
import org.apache.spark.sql.catalyst.plans.physical.BroadcastMode
import org.apache.spark.sql.catalyst.trees.TreeNode
import org.apache.spark.sql.execution._
import org.apache.spark.sql.execution.adaptive.{BroadcastQueryStageExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.command.RunnableCommand
import org.apache.spark.sql.execution.datasources.{FileIndex, FilePartition, FileScanRDD, HadoopFsRelation, InMemoryFileIndex, PartitionDirectory, PartitionedFile}
import org.apache.spark.sql.execution.exchange.ReusedExchangeExec
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, ShuffledHashJoinExec, SortMergeJoinExec}
import org.apache.spark.sql.execution.python.{AggregateInPandasExec, ArrowEvalPythonExec, FlatMapGroupsInPandasExec, MapInPandasExec, WindowInPandasExec}
import org.apache.spark.sql.execution.window.WindowExecBase
import org.apache.spark.sql.internal.SQLConf

/**
 * This class contains the default implementation for most shim methods and should be compiled
 * against the lowest supported Spark version. As support for the oldest Spark version is
 * abandoned, this class should move to the next oldest supported version and overrides from
 * that version should be folded into here. Any shim methods that are implemented only in the
 * updated base version can then be removed from the shim interface.
 */
abstract class SparkBaseShims extends PluginShims {

  override def parquetRebaseReadKey: String =
    SQLConf.LEGACY_PARQUET_REBASE_MODE_IN_READ.key
  override def parquetRebaseWriteKey: String =
    SQLConf.LEGACY_PARQUET_REBASE_MODE_IN_WRITE.key
  override def avroRebaseReadKey: String =
    SQLConf.LEGACY_AVRO_REBASE_MODE_IN_READ.key
  override def avroRebaseWriteKey: String =
    SQLConf.LEGACY_AVRO_REBASE_MODE_IN_WRITE.key
  override def parquetRebaseRead(conf: SQLConf): String =
    conf.getConf(SQLConf.LEGACY_PARQUET_REBASE_MODE_IN_READ)
  override def parquetRebaseWrite(conf: SQLConf): String =
    conf.getConf(SQLConf.LEGACY_PARQUET_REBASE_MODE_IN_WRITE)

  override def v1RepairTableCommand(tableName: TableIdentifier): RunnableCommand =
    AlterTableRecoverPartitionsCommand(tableName)

  override def getGpuBroadcastExchangeExec(
      mode: BroadcastMode,
      child: SparkPlan): GpuBroadcastExchangeExecBase = {
    GpuBroadcastExchangeExec(mode, child)
  }

  override def getGpuShuffleExchangeExec(
      queryStage: ShuffleQueryStageExec): GpuShuffleExchangeExecBase = {
    queryStage.shuffle.asInstanceOf[GpuShuffleExchangeExecBase]
  }

  override def isWindowFunctionExec(plan: SparkPlan): Boolean = plan.isInstanceOf[WindowExecBase]

  override def getExecs: Map[Class[_ <: SparkPlan], ExecRule[_ <: SparkPlan]] = {
    Seq(
      GpuOverrides.exec[WindowInPandasExec](
      "The backend for Window Aggregation Pandas UDF, Accelerates the data transfer between" +
          " the Java process and the Python process. It also supports scheduling GPU resources" +
          " for the Python process when enabled. For now it only supports row based window frame.",
        ExecChecks(
          (TypeSig.commonCudfTypes + TypeSig.ARRAY).nested(TypeSig.commonCudfTypes),
          TypeSig.all),
        (winPy, conf, p, r) => new GpuWindowInPandasExecMetaBase(winPy, conf, p, r) {
          override val windowExpressions: Seq[BaseExprMeta[NamedExpression]] =
            winPy.windowExpression.map(GpuOverrides.wrapExpr(_, conf, Some(this)))

          override def convertToGpu(): GpuExec = {
            GpuWindowInPandasExec(
              windowExpressions.map(_.convertToGpu()),
              partitionSpec.map(_.convertToGpu()),
              orderSpec.map(_.convertToGpu().asInstanceOf[SortOrder]),
              childPlans.head.convertIfNeeded()
            )
          }
        }).disabledByDefault("it only supports row based frame for now"),
      GpuOverrides.exec[FileSourceScanExec](
        "Reading data from files, often from Hive tables",
        ExecChecks((TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.STRUCT + TypeSig.MAP +
            TypeSig.ARRAY + TypeSig.DECIMAL).nested(), TypeSig.all),
        (fsse, conf, p, r) => new SparkPlanMeta[FileSourceScanExec](fsse, conf, p, r) {

          // partition filters and data filters are not run on the GPU
          override val childExprs: Seq[ExprMeta[_]] = Seq.empty

          override def tagPlanForGpu(): Unit = GpuFileSourceScanExec.tagSupport(this)

          override def convertToGpu(): GpuExec = {
            val sparkSession = wrapped.relation.sparkSession
            val options = wrapped.relation.options

            val location = replaceWithAlluxioPathIfNeeded(
              conf,
              wrapped.relation,
              wrapped.partitionFilters,
              wrapped.dataFilters)

            val newRelation = HadoopFsRelation(
              location,
              wrapped.relation.partitionSchema,
              wrapped.relation.dataSchema,
              wrapped.relation.bucketSpec,
              GpuFileSourceScanExec.convertFileFormat(wrapped.relation.fileFormat),
              options)(sparkSession)

            GpuFileSourceScanExec(
              newRelation,
              wrapped.output,
              wrapped.requiredSchema,
              wrapped.partitionFilters,
              wrapped.optionalBucketSet,
              None,
              wrapped.dataFilters,
              wrapped.tableIdentifier)(conf)
          }
        }),
      GpuOverrides.exec[SortMergeJoinExec](
        "Sort merge join, replacing with shuffled hash join",
        ExecChecks((TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL + TypeSig.ARRAY +
            TypeSig.STRUCT).nested(TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL
        ), TypeSig.all),
        (join, conf, p, r) => new GpuSortMergeJoinMeta(join, conf, p, r)),
      GpuOverrides.exec[BroadcastHashJoinExec](
        "Implementation of join using broadcast data",
        ExecChecks((TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL + TypeSig.ARRAY +
            TypeSig.STRUCT).nested(TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL
        ), TypeSig.all),
        (join, conf, p, r) => new GpuBroadcastHashJoinMeta(join, conf, p, r)),
      GpuOverrides.exec[ShuffledHashJoinExec](
        "Implementation of join using hashed shuffled data",
        ExecChecks((TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL + TypeSig.ARRAY +
            TypeSig.STRUCT).nested(TypeSig.commonCudfTypes + TypeSig.NULL + TypeSig.DECIMAL
        ), TypeSig.all),
        (join, conf, p, r) => new GpuShuffledHashJoinMeta(join, conf, p, r)),
      GpuOverrides.exec[ArrowEvalPythonExec](
        "The backend of the Scalar Pandas UDFs. Accelerates the data transfer between the" +
            " Java process and the Python process. It also supports scheduling GPU resources" +
            " for the Python process when enabled",
        ExecChecks(
          (TypeSig.commonCudfTypes + TypeSig.ARRAY + TypeSig.STRUCT).nested(),
          TypeSig.all),
        (e, conf, p, r) =>
          new SparkPlanMeta[ArrowEvalPythonExec](e, conf, p, r) {
            val udfs: Seq[BaseExprMeta[PythonUDF]] =
              e.udfs.map(GpuOverrides.wrapExpr(_, conf, Some(this)))
            val resultAttrs: Seq[BaseExprMeta[Attribute]] =
              e.resultAttrs.map(GpuOverrides.wrapExpr(_, conf, Some(this)))
            override val childExprs: Seq[BaseExprMeta[_]] = udfs ++ resultAttrs

            override def replaceMessage: String = "partially run on GPU"
            override def noReplacementPossibleMessage(reasons: String): String =
              s"cannot run even partially on the GPU because $reasons"

            override def convertToGpu(): GpuExec =
              GpuArrowEvalPythonExec(udfs.map(_.convertToGpu()).asInstanceOf[Seq[GpuPythonUDF]],
                resultAttrs.map(_.convertToGpu()).asInstanceOf[Seq[Attribute]],
                childPlans.head.convertIfNeeded(),
                e.evalType)
          }),
      GpuOverrides.exec[MapInPandasExec](
        "The backend for Map Pandas Iterator UDF. Accelerates the data transfer between the" +
            " Java process and the Python process. It also supports scheduling GPU resources" +
            " for the Python process when enabled.",
        ExecChecks((TypeSig.commonCudfTypes + TypeSig.ARRAY + TypeSig.STRUCT).nested(),
          TypeSig.all),
        (mapPy, conf, p, r) => new GpuMapInPandasExecMeta(mapPy, conf, p, r)),
      GpuOverrides.exec[FlatMapGroupsInPandasExec](
        "The backend for Flat Map Groups Pandas UDF, Accelerates the data transfer between the" +
            " Java process and the Python process. It also supports scheduling GPU resources" +
            " for the Python process when enabled.",
        ExecChecks(TypeSig.commonCudfTypes, TypeSig.all),
        (flatPy, conf, p, r) => new GpuFlatMapGroupsInPandasExecMeta(flatPy, conf, p, r)),
      GpuOverrides.exec[AggregateInPandasExec](
        "The backend for an Aggregation Pandas UDF, this accelerates the data transfer between" +
            " the Java process and the Python process. It also supports scheduling GPU resources" +
            " for the Python process when enabled.",
        ExecChecks(TypeSig.commonCudfTypes, TypeSig.all),
        (aggPy, conf, p, r) => new GpuAggregateInPandasExecMeta(aggPy, conf, p, r))
    ).map(r => (r.getClassFor.asSubclass(classOf[SparkPlan]), r)).toMap
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
        (a, conf, p, r) => new QuaternaryExprMeta[RegExpReplace](a, conf, p, r) {
          override def tagExprForGpu(): Unit = {
            if (GpuOverrides.isNullOrEmptyOrRegex(a.regexp)) {
              willNotWorkOnGpu(
                "Only non-null, non-empty String literals that are not regex patterns " +
                    "are supported by RegExpReplace on the GPU")
            }
          }

          // TODO: implement replace from non-0 position
          override def convertToGpu(lhs: Expression, regexp: Expression,
              rep: Expression, pos: Expression): GpuExpression = GpuStringReplace(lhs, regexp, rep)
        })
    ).map(r => (r.getClassFor.asSubclass(classOf[Expression]), r)).toMap
  }

  override def getPartitionFileNames(
      partitions: Seq[PartitionDirectory]): Seq[String] = {
    val files = partitions.flatMap(partition => partition.files)
    files.map(_.getPath.getName)
  }

  override def getPartitionFileStatusSize(partitions: Seq[PartitionDirectory]): Long = {
    partitions.map(_.files.map(_.getLen).sum).sum
  }

  override def getPartitionedFiles(
      partitions: Array[PartitionDirectory]): Array[PartitionedFile] = {
    partitions.flatMap { p =>
      p.files.map { f =>
        PartitionedFileUtil.getPartitionedFile(f, f.getPath, p.values)
      }
    }
  }

  override def getPartitionSplitFiles(
      partitions: Array[PartitionDirectory],
      maxSplitBytes: Long,
      relation: HadoopFsRelation): Array[PartitionedFile] = {
    partitions.flatMap { partition =>
      partition.files.flatMap { file =>
        // getPath() is very expensive so we only want to call it once in this block:
        val filePath = file.getPath
        val isSplitable = relation.fileFormat.isSplitable(
          relation.sparkSession, relation.options, filePath)
        PartitionedFileUtil.splitFiles(
          sparkSession = relation.sparkSession,
          file = file,
          filePath = filePath,
          isSplitable = isSplitable,
          maxSplitBytes = maxSplitBytes,
          partitionValues = partition.values
        )
      }
    }
  }

  override def getFileScanRDD(
      sparkSession: SparkSession,
      readFunction: PartitionedFile => Iterator[InternalRow],
      filePartitions: Seq[FilePartition]): RDD[InternalRow] = {
    new FileScanRDD(sparkSession, readFunction, filePartitions)
  }

  override def createFilePartition(index: Int, files: Array[PartitionedFile]): FilePartition = {
    FilePartition(index, files)
  }

  override def replaceWithAlluxioPathIfNeeded(
      conf: RapidsConf,
      relation: HadoopFsRelation,
      partitionFilters: Seq[Expression],
      dataFilters: Seq[Expression]): FileIndex = {

    val alluxioPathsReplace: Option[Seq[String]] = conf.getAlluxioPathsToReplace

    if (alluxioPathsReplace.isDefined) {
      // alluxioPathsReplace: Seq("key->value", "key1->value1")
      // turn the rules to the Map with eg
      // { s3:/foo -> alluxio://0.1.2.3:19998/foo,
      //   gs:/bar -> alluxio://0.1.2.3:19998/bar,
      //   /baz -> alluxio://0.1.2.3:19998/baz }
      val replaceMapOption = alluxioPathsReplace.map(rules => {
        rules.map(rule => {
          val split = rule.split("->")
          if (split.size == 2) {
            split(0).trim -> split(1).trim
          } else {
            throw new IllegalArgumentException(s"Invalid setting for " +
              s"${RapidsConf.ALLUXIO_PATHS_REPLACE.key}")
          }
        }).toMap
      })

      replaceMapOption.map(replaceMap => {

        def isDynamicPruningFilter(e: Expression): Boolean =
          e.find(_.isInstanceOf[PlanExpression[_]]).isDefined

        val partitionDirs = relation.location.listFiles(
          partitionFilters.filterNot(isDynamicPruningFilter), dataFilters)

        // replacement func to check if the file path is prefixed with the string user configured
        // if yes, replace it
        val replaceFunc = (f: Path) => {
          val pathStr = f.toString
          val matchedSet = replaceMap.keySet.filter(reg => pathStr.startsWith(reg))
          if (matchedSet.size > 1) {
            // never reach here since replaceMap is a Map
            throw new IllegalArgumentException(s"Found ${matchedSet.size} same replacing rules " +
              s"from ${RapidsConf.ALLUXIO_PATHS_REPLACE.key} which requires only 1 rule for each " +
              s"file path")
          } else if (matchedSet.size == 1) {
            new Path(pathStr.replaceFirst(matchedSet.head, replaceMap(matchedSet.head)))
          } else {
            f
          }
        }

        // replace all of input files
        val inputFiles: Seq[Path] = partitionDirs.flatMap(partitionDir => {
          replacePartitionDirectoryFiles(partitionDir, replaceFunc)
        })

        // replace all of rootPaths which are already unique
        val rootPaths = relation.location.rootPaths.map(replaceFunc)

        val parameters: Map[String, String] = relation.options

        // infer PartitionSpec
        val partitionSpec = GpuPartitioningUtils.inferPartitioning(
          relation.sparkSession,
          rootPaths,
          inputFiles,
          parameters,
          Option(relation.dataSchema),
          replaceFunc)

        // generate a new InMemoryFileIndex holding paths with alluxio schema
        new InMemoryFileIndex(
          relation.sparkSession,
          inputFiles,
          parameters,
          Option(relation.dataSchema),
          userSpecifiedPartitionSpec = Some(partitionSpec))
      }).getOrElse(relation.location)

    } else {
      relation.location
    }
  }

  override def replacePartitionDirectoryFiles(partitionDir: PartitionDirectory,
      replaceFunc: Path => Path): Seq[Path] = {
    partitionDir.files.map(f => replaceFunc(f.getPath))
  }

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

  override def hasAliasQuoteFix: Boolean = false
}
