/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.
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

package org.apache.spark.sql.rapids

import java.util.UUID
import ai.rapids.cudf.{Cuda, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.{RapidsBufferCatalog, RapidsBufferId, RapidsBufferStore, RapidsBufferWithMeta, RapidsDeviceMemoryStore, SpillableColumnarBatchImpl, StorageTier}
import com.nvidia.spark.rapids.StorageTier.StorageTier
import org.scalatest.funsuite.AnyFunSuite
import org.apache.spark.sql.types.{DataType, IntegerType}
import org.apache.spark.sql.vectorized.ColumnarBatch
import org.apache.spark.storage.TempLocalBlockId
import com.nvidia.spark.rapids.RapidsMemoryBuffer
import com.nvidia.spark.rapids.RapidsBuffer
import com.nvidia.spark.rapids.RapidsBufferCopyIterator

class SpillableColumnarBatchSuite extends AnyFunSuite {

  test("close updates catalog") {
    withResource(new RapidsDeviceMemoryStore) { deviceStore =>
      val id = TempSpillBufferId(0, TempLocalBlockId(new UUID(1, 2)))
      withResource(new RapidsBufferCatalog(deviceStore)) { catalog =>
        val mockBuffer = new MockBuffer(id, catalog)
        val rmb = new RapidsMemoryBuffer(id)
        rmb.initialize(mockBuffer, StorageTier.DEVICE)
        val oldBufferCount = catalog.numBuffers
        catalog.registerNewBuffer(rmb)
        val handle = catalog.makeNewHandle(id, -1)
        assertResult(oldBufferCount + 1)(catalog.numBuffers)
        val spillableBatch = new SpillableColumnarBatchImpl(
          handle,
          5,
          Array[DataType](IntegerType))
        spillableBatch.close()
        assertResult(oldBufferCount)(catalog.numBuffers)
      }
    }
  }

  class MockBuffer(override val id: RapidsBufferId, catalog: RapidsBufferCatalog)
    extends RapidsBuffer {
    override val base: RapidsMemoryBuffer = null
    // TODO: fixme: why is this RBBWM
    override def getCopyIterator(stream: Cuda.Stream): RapidsBufferCopyIterator = null
    override def copyTo(
      store: RapidsBufferStore, stream: Cuda.Stream): RapidsBuffer = null
    override val memoryUsedBytes: Long = 123
    override val storageTier: StorageTier = StorageTier.DEVICE
    override def addReference(): Boolean = true
    override def free(): Unit = {}
    override def getSpillPriority: Long = 0
    override def close(): Unit = {}
    override def setSpillPriority(newPriority: Long): Unit = {}
  }
}
