/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.
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

import java.io.{ObjectInputStream, ObjectOutputStream}

import ai.rapids.cudf.{JCudfSerialization, Table}
import com.nvidia.spark.rapids.Arm.withResource
import org.apache.commons.io.output.ByteArrayOutputStream
import org.apache.commons.lang3.SerializationUtils
import org.scalatest.BeforeAndAfterAll
import org.scalatest.funsuite.AnyFunSuite

import org.apache.spark.sql.catalyst.expressions.AttributeReference
import org.apache.spark.sql.rapids.execution.{SerializeBatchDeserializeHostBuffer, SerializeConcatHostBuffersDeserializeBatch}
import org.apache.spark.sql.types.{DoubleType, FloatType, IntegerType, StringType}
import org.apache.spark.sql.vectorized.ColumnarBatch

class SerializationSuite extends AnyFunSuite
  with BeforeAndAfterAll {

  override def beforeAll(): Unit = {
    RapidsBufferCatalog.setDeviceStorage(new RapidsDeviceMemoryStore())
  }

  private def buildBatch(): ColumnarBatch = {
    withResource(new Table.TestBuilder()
        .column(5, null.asInstanceOf[java.lang.Integer], 3, 1, 1, 1, 1, 1, 1, 1)
        .column("five", "two", null, null, "one", "one", "one", "one", "one", "one")
        .column(5.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        .build()) { table =>
      GpuColumnVector.from(table, Array(IntegerType, StringType, DoubleType))
    }
  }

  /**
   * Creates an empty batch: has columns but numRows = 0
   */
  private def buildEmptyBatch(): ColumnarBatch = {
    GpuColumnVector.emptyBatchFromTypes(Array(IntegerType, FloatType))
  }

  /**
   * Creates a completely empty batch: no columns nor rows
   */
  private def buildEmptyBatchNoCols(): ColumnarBatch = {
    GpuColumnVector.emptyBatchFromTypes(Array.empty)
  }

  /**
   * Creates a "just rows" batch: no columns but numRows > 0.
   * Seen with a no-condition cross join followed by a count
   */
  private def buildJustRowsBatch(): ColumnarBatch = {
    new ColumnarBatch(Array.empty, 1234)
  }

  /**
   * Creates a "just rows" batch: no columns and numrows == 0
   * Seen with a no-condition cross join followed by a count
   */
  private def buildJustRowsBatchZeroRows(): ColumnarBatch = {
    new ColumnarBatch(Array.empty, 0)
  }

  private def createDeserializedHostBuffer(
      batch: ColumnarBatch): SerializeBatchDeserializeHostBuffer = {
    withResource(new SerializeBatchDeserializeHostBuffer(batch)) { obj =>
      // Return a deserialized form of the object as if it was read on the driver
      SerializationUtils.clone(obj)
    }
  }

  test("SerializeConcatHostBuffersDeserializeBatch driver serialize after deserialize") {
    val hostBatch = withResource(buildBatch()) { gpuBatch =>
      val attrs = GpuColumnVector.extractTypes(gpuBatch).map(t => AttributeReference("", t)())
      withResource(createDeserializedHostBuffer(gpuBatch)) { buffer =>
        val hostConcatResult =
          JCudfSerialization.concatToHostBuffer(Array(buffer.header), Array(buffer.buffer))
        new SerializeConcatHostBuffersDeserializeBatch(
          hostConcatResult,
          attrs,
          hostConcatResult.getTableHeader.getNumRows,
          hostConcatResult.getTableHeader.getDataLen)
      }
    }
    try {
      // clone via serialization without manifesting the GPU batch
      val clonedObj = SerializationUtils.clone(hostBatch)
      try {
        assert(clonedObj.data != null)
        assertResult(hostBatch.dataSize)(clonedObj.dataSize)
        assertResult(hostBatch.numRows)(clonedObj.numRows)
        // try to clone it again from the cloned object
        SerializationUtils.clone(clonedObj).closeInternal()
      } finally {
        clonedObj.closeInternal()
      }
    } finally {
      hostBatch.closeInternal()
    }
  }

  test("SerializeConcatHostBuffersDeserializeBatch driver obtain hostBatch") {
    val hostBatch = withResource(buildBatch()) { gpuBatch =>
      val attrs = GpuColumnVector.extractTypes(gpuBatch).map(t => AttributeReference("", t)())
      withResource(createDeserializedHostBuffer(gpuBatch)) { buffer =>
        val hostConcatResult =
          JCudfSerialization.concatToHostBuffer(Array(buffer.header), Array(buffer.buffer))
        new SerializeConcatHostBuffersDeserializeBatch(
          hostConcatResult,
          attrs,
          hostConcatResult.getTableHeader.getNumRows,
          hostConcatResult.getTableHeader.getDataLen)
      }
    }
    try {
      withResource(hostBatch.hostBatch) { hostBatch1 =>
        val clonedObj = SerializationUtils.clone(hostBatch)
        withResource(clonedObj.hostBatch) { hostBatch2 =>
          TestUtils.compareBatches(hostBatch1, hostBatch2)
        }
      }
    } finally {
      hostBatch.closeInternal()
    }
  }

  test("SerializeConcatHostBuffersDeserializeBatch executor ser/deser empty batch") {
    withResource(Seq(buildEmptyBatch(), buildEmptyBatchNoCols())) { batches =>
      batches.foreach { gpuExpected =>
        val numRows = gpuExpected.numRows()
        val attrs = GpuColumnVector.extractTypes(gpuExpected).map(t => AttributeReference("", t)())
        val hostBatch = withResource(createDeserializedHostBuffer(gpuExpected)) { buffer =>
          val hostConcatResult =
            JCudfSerialization.concatToHostBuffer(Array(buffer.header), Array(buffer.buffer))
          new SerializeConcatHostBuffersDeserializeBatch(hostConcatResult, attrs, numRows, 0)
        }
        try {
          withResource(hostBatch.batch.getColumnarBatch()) { gpuBatch =>
            TestUtils.compareBatches(gpuExpected, gpuBatch)
          }
          // clone via serialization after manifesting the GPU batch
          val clonedObj = SerializationUtils.clone(hostBatch)
          try {
            withResource(clonedObj.batch.getColumnarBatch()) { gpuClonedBatch =>
              TestUtils.compareBatches(gpuExpected, gpuClonedBatch)
            }
            // try to clone it again from the cloned object
            SerializationUtils.clone(clonedObj).closeInternal()
          } finally {
            clonedObj.closeInternal()
          }
        } finally {
          hostBatch.closeInternal()
        }
      }
    }
  }

  test("SerializeConcatHostBuffersDeserializeBatch executor ser/deser just rows") {
    withResource(Seq(buildJustRowsBatch(), buildJustRowsBatchZeroRows())) { batches =>
      batches.foreach { gpuExpected =>
        val attrs = GpuColumnVector.extractTypes(gpuExpected).map(t => AttributeReference("", t)())
        val hostBatch =
          new SerializeConcatHostBuffersDeserializeBatch(
            null, attrs, gpuExpected.numRows(), 0L)
        try {
          withResource(hostBatch.batch.getColumnarBatch()) { gpuBatch =>
            TestUtils.compareBatches(gpuExpected, gpuBatch)
          }
          // clone via serialization after manifesting the GPU batch
          val clonedObj = SerializationUtils.clone(hostBatch)
          try {
            withResource(clonedObj.batch.getColumnarBatch()) { gpuClonedBatch =>
              TestUtils.compareBatches(gpuExpected, gpuClonedBatch)
            }
            // try to clone it again from the cloned object
            SerializationUtils.clone(clonedObj).closeInternal()
          } finally {
            clonedObj.closeInternal()
          }
        } finally {
          hostBatch.closeInternal()
        }
      }
    }
  }

  test("SerializeConcatHostBuffersDeserializeBatch executor serialize after deserialize") {
    withResource(buildBatch()) { gpuExpected =>
      val attrs = GpuColumnVector.extractTypes(gpuExpected).map(t => AttributeReference("", t)())
      val hostBatch = withResource(createDeserializedHostBuffer(gpuExpected)) { buffer =>
        val hostConcatResult =
          JCudfSerialization.concatToHostBuffer(Array(buffer.header), Array(buffer.buffer))
        new SerializeConcatHostBuffersDeserializeBatch(
          hostConcatResult,
          attrs,
          hostConcatResult.getTableHeader.getNumRows,
          hostConcatResult.getTableHeader.getDataLen)
      }
      try {
        withResource(hostBatch.batch.getColumnarBatch()) { gpuBatch =>
          TestUtils.compareBatches(gpuExpected, gpuBatch)
        }
        // clone via serialization after manifesting the GPU batch
        val clonedObj = SerializationUtils.clone(hostBatch)
        try {
          withResource(clonedObj.batch.getColumnarBatch()) { gpuClonedBatch =>
            TestUtils.compareBatches(gpuExpected, gpuClonedBatch)
          }
          // try to clone it again from the cloned object
          SerializationUtils.clone(clonedObj).closeInternal()
        } finally {
          clonedObj.closeInternal()
        }
      } finally {
        hostBatch.closeInternal()
      }
    }
  }

  test("SerializeConcatHostBuffersDeserializeBatch reuse after spill") {
    withResource(buildBatch()) { gpuExpected =>
      val attrs = GpuColumnVector.extractTypes(gpuExpected).map(t => AttributeReference("", t)())
      val hostBatch = withResource(createDeserializedHostBuffer(gpuExpected)) { buffer =>
        val hostConcatResult =
          JCudfSerialization.concatToHostBuffer(Array(buffer.header), Array(buffer.buffer))
        new SerializeConcatHostBuffersDeserializeBatch(
          hostConcatResult,
          attrs,
          hostConcatResult.getTableHeader.getNumRows,
          hostConcatResult.getTableHeader.getDataLen)
      }
      try {
        // spill first thing (we didn't materialize it in the executor)
        val baos = new ByteArrayOutputStream()
        val oos = new ObjectOutputStream(baos)
        hostBatch.doWriteObject(oos)

        val inputStream = new ObjectInputStream(baos.toInputStream)
        hostBatch.doReadObject(inputStream)

        // use it now
        withResource(hostBatch.batch.getColumnarBatch()) { gpuBatch =>
          TestUtils.compareBatches(gpuExpected, gpuBatch)
        }
      } finally {
        hostBatch.closeInternal()
      }
    }
  }

  test("SerializeConcatHostBuffersDeserializeBatch cloned use after spill") {
    withResource(buildBatch()) { gpuExpected =>
      val attrs = GpuColumnVector.extractTypes(gpuExpected).map(t => AttributeReference("", t)())
      val hostBatch = withResource(createDeserializedHostBuffer(gpuExpected)) { buffer =>
        val hostConcatResult =
          JCudfSerialization.concatToHostBuffer(Array(buffer.header), Array(buffer.buffer))
        new SerializeConcatHostBuffersDeserializeBatch(
          hostConcatResult,
          attrs,
          hostConcatResult.getTableHeader.getNumRows,
          hostConcatResult.getTableHeader.getDataLen)
      }
      try {
        // spill first thing (we didn't materialize it in the executor)
        val baos = new ByteArrayOutputStream()
        SerializationUtils.serialize(hostBatch, baos)

        val materialized = SerializationUtils
            .deserialize[SerializeConcatHostBuffersDeserializeBatch](baos.toInputStream)

        // use it now
        withResource(materialized.batch.getColumnarBatch()) { gpuBatch =>
          TestUtils.compareBatches(gpuExpected, gpuBatch)
        }
        materialized.closeInternal()
      } finally {
        hostBatch.closeInternal()
      }
    }
  }

  test("SerializeConcatHostBuffersDeserializeBatch reuse materialized after spill") {
    withResource(buildBatch()) { gpuExpected =>
      val attrs = GpuColumnVector.extractTypes(gpuExpected).map(t => AttributeReference("", t)())
      val hostBatch = withResource(createDeserializedHostBuffer(gpuExpected)) { buffer =>
        val hostConcatResult =
          JCudfSerialization.concatToHostBuffer(Array(buffer.header), Array(buffer.buffer))
        new SerializeConcatHostBuffersDeserializeBatch(
          hostConcatResult,
          attrs,
          hostConcatResult.getTableHeader.getNumRows,
          hostConcatResult.getTableHeader.getDataLen)
      }
      try {
        withResource(hostBatch.batch.getColumnarBatch()) { gpuBatch =>
          TestUtils.compareBatches(gpuExpected, gpuBatch)
        }
        val baos = new ByteArrayOutputStream()
        val oos = new ObjectOutputStream(baos)
        hostBatch.doWriteObject(oos)

        val inputStream = new ObjectInputStream(baos.toInputStream)
        hostBatch.doReadObject(inputStream)
        withResource(hostBatch.batch.getColumnarBatch()) { gpuBatch =>
          TestUtils.compareBatches(gpuExpected, gpuBatch)
        }
      } finally {
        hostBatch.closeInternal()
      }
    }
  }
}
