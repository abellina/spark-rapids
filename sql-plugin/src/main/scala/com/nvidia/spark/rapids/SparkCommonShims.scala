package com.nvidia.spark.rapids

import java.util.UUID

import org.apache.hadoop.fs.Path

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.catalyst.expressions.{Attribute, Expression, NamedExpression, PlanExpression, PythonUDF, SortOrder, UnaryMinus}
import org.apache.spark.sql.execution.datasources.{FileIndex, FilePartition, FileScanRDD, HadoopFsRelation, InMemoryFileIndex, PartitionDirectory, PartitionedFile}
import org.apache.spark.sql.execution.python.{AggregateInPandasExec, ArrowEvalPythonExec, FlatMapGroupsInPandasExec, MapInPandasExec, WindowInPandasExec}
import org.apache.spark.sql.execution.{FileSourceScanExec, PartitionedFileUtil, SparkPlan}
import org.apache.spark.sql.execution.joins.{BroadcastHashJoinExec, ShuffledHashJoinExec, SortMergeJoinExec}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.rapids.execution.python.{GpuAggregateInPandasExecMeta, GpuArrowEvalPythonExec, GpuFlatMapGroupsInPandasExecMeta, GpuMapInPandasExecMeta, GpuPythonUDF, GpuWindowInPandasExecMetaBase}
import org.apache.spark.sql.rapids.GpuFileSourceScanExec
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.plans.logical.Statistics
import org.apache.spark.sql.catalyst.plans.physical.BroadcastMode
import org.apache.spark.sql.execution.adaptive.{BroadcastQueryStageExec, ShuffleQueryStageExec}
import org.apache.spark.sql.execution.datasources.rapids.GpuPartitioningUtils
import org.apache.spark.sql.execution.exchange.{BroadcastExchangeLike, ReusedExchangeExec}
import org.apache.spark.sql.execution.window.WindowExecBase
import org.apache.spark.sql.rapids.execution.{GpuBroadcastExchangeExecBase, GpuBroadcastExchangeExecBaseWithFuture, GpuShuffleExchangeExecBase}

abstract class SparkCommonShims extends SparkShims {
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
            GpuDefaultWindowInPandasExec(
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

  override def createFilePartition(index: Int, files: Array[PartitionedFile]): FilePartition = {
    FilePartition(index, files)
  }

  override def getFileScanRDD(
      sparkSession: SparkSession,
      readFunction: PartitionedFile => Iterator[InternalRow],
      filePartitions: Seq[FilePartition]): RDD[InternalRow] = {
    new FileScanRDD(sparkSession, readFunction, filePartitions)
  }

  override def getGpuBroadcastExchangeExec(
      mode: BroadcastMode,
      child: SparkPlan): GpuBroadcastExchangeExecBase = {
    GpuBroadcastExchangeExec(mode, child)
  }

  override def getGpuBroadcastExchangeExecBase(buildPlan: SparkPlan) = {
    buildPlan match {
      case BroadcastQueryStageExec(_, gpu: GpuBroadcastExchangeExec) => gpu
      case BroadcastQueryStageExec(_, reused: ReusedExchangeExec) =>
        reused.child.asInstanceOf[GpuBroadcastExchangeExec]
      case gpu: GpuBroadcastExchangeExec => gpu
      case reused: ReusedExchangeExec => reused.child.asInstanceOf[GpuBroadcastExchangeExec]
    }
  }

  override def getGpuShuffleExchangeExec(
      queryStage: ShuffleQueryStageExec): GpuShuffleExchangeExecBase = {
    queryStage.shuffle.asInstanceOf[GpuShuffleExchangeExecBase]
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

  override def hasAliasQuoteFix: Boolean = false

  override def hasCastFloatTimestampUpcast: Boolean = false

  override def isWindowFunctionExec(plan: SparkPlan): Boolean = plan.isInstanceOf[WindowExecBase]

  override def replacePartitionDirectoryFiles(partitionDir: PartitionDirectory,
      replaceFunc: Path => Path): Seq[Path] = {
    partitionDir.files.map(f => replaceFunc(f.getPath))
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
}

case class GpuBroadcastExchangeExec(
    override val mode: BroadcastMode,
    child: SparkPlan) extends GpuBroadcastExchangeExecBaseWithFuture(mode, child)
    with BroadcastExchangeLike {

  override def runId: UUID = _runId

  override def runtimeStatistics: Statistics = {
    Statistics(
      sizeInBytes = metrics("dataSize").value,
      rowCount = Some(metrics(GpuMetric.NUM_OUTPUT_ROWS).value))
  }

  override def doCanonicalize(): SparkPlan = {
    GpuBroadcastExchangeExec(mode.canonicalized, child.canonicalized)
  }
}
