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

package com.nvidia.spark.rapids.spill

import java.io.File

import ai.rapids.cudf.{Cuda, DeviceMemoryBuffer, MemoryBuffer}
import com.nvidia.spark.rapids.{Arm, GpuMetric}
import com.nvidia.spark.rapids.format.TableMeta
import org.mockito.ArgumentMatchers.any
import org.mockito.Mockito._
import org.scalatest.FunSuite
import org.scalatest.mockito.MockitoSugar

import org.apache.spark.sql.rapids.RapidsDiskBlockManager

class RapidsBufferCatalogSuite extends FunSuite with MockitoSugar with Arm {
  test("lookup unknown buffer") {
    val catalog = new RapidsBufferCatalog
    val bufferId = new RapidsBufferId {
      override val tableId: Int = 10

      override def getDiskPath(m: RapidsDiskBlockManager): File = null
    }
    val bufferHandle = new RapidsBufferHandle {
      override val id: RapidsBufferId = bufferId

      override def setSpillPriority(newPriority: Long): Unit = {}

      override def close(): Unit = {}
    }

    assertThrows[NoSuchElementException](catalog.acquireBuffer(bufferHandle))
    assertThrows[NoSuchElementException](catalog.getBufferMeta(bufferId))
  }

  test("buffer double register throws") {
    val catalog = new RapidsBufferCatalog
    val bufferId = MockBufferId(5)
    val buffer = mockBuffer(bufferId)
    catalog.registerNewBuffer(buffer)
    val buffer2 = mockBuffer(bufferId)
    assertThrows[DuplicateBufferException](
      catalog.registerNewBuffer(buffer2))
  }

  test("a second handle prevents buffer to be removed") {
    val catalog = new RapidsBufferCatalog
    val bufferId = MockBufferId(5)
    val buffer = mockBuffer(bufferId)
    val handle1 = catalog.registerNewBuffer(buffer)
    val handle2 =
      catalog.makeNewHandle(bufferId, -1, RapidsBuffer.defaultSpillCallback)

    handle1.close()

    // this does not throw
    catalog.acquireBuffer(handle2).close()
    // actually this doesn't throw either
    catalog.acquireBuffer(handle1).close()

    handle2.close()

    assertThrows[NoSuchElementException](catalog.acquireBuffer(handle1))
    assertThrows[NoSuchElementException](catalog.acquireBuffer(handle2))
  }

  test("spill priorities are updated as handles are registered and unregistered") {
    val catalog = new RapidsBufferCatalog
    val bufferId = MockBufferId(5)
    val buffer = mockBuffer(bufferId, initialPriority = -1)
    val handle1 = catalog.registerNewBuffer(buffer)
    withResource(catalog.acquireBuffer(handle1)) { buff =>
      assertResult(-1)(buff.getSpillPriority)
    }
    val handle2 =
      catalog.makeNewHandle(bufferId, 0, RapidsBuffer.defaultSpillCallback)
    withResource(catalog.acquireBuffer(handle2)) { buff =>
      assertResult(0)(buff.getSpillPriority)
    }

    // removing the lower priority handle, keeps the high priority spill
    handle1.close()
    withResource(catalog.acquireBuffer(handle2)) { buff =>
      assertResult(0)(buff.getSpillPriority)
    }

    // adding a lower priority -1000 handle keeps the high priority (0) spill
    val handle3 =
      catalog.makeNewHandle(bufferId, -1000, RapidsBuffer.defaultSpillCallback)
    withResource(catalog.acquireBuffer(handle3)) { buff =>
      assertResult(0)(buff.getSpillPriority)
    }

    // removing the high priority spill (0) brings us down to the
    // low priority that is remaining
    handle2.close()
    withResource(catalog.acquireBuffer(handle2)) { buff =>
      assertResult(-1000)(buff.getSpillPriority)
    }

    handle3.close()
  }

  test("spill callbacks are updated as handles are registered and unregistered") {
    val catalog = new RapidsBufferCatalog
    val bufferId = MockBufferId(5)
    val buffer = mockBuffer(bufferId, initialPriority = -1)
    val handle1 = catalog.registerNewBuffer(buffer)
    withResource(catalog.acquireBuffer(handle1)) { buff =>
      assertResult(null)(buff.getSpillCallback)
    }
    val handle2 =
      catalog.makeNewHandle(bufferId, 0, RapidsBuffer.defaultSpillCallback)
    withResource(catalog.acquireBuffer(handle2)) { buff =>
      assertResult(RapidsBuffer.defaultSpillCallback)(buff.getSpillCallback)
    }

    // adding a new handle puts a new callback in front, that's the new callback
    val mySpillCallback = new SpillMetricsCallback {
      override def apply(from: StorageTier, to: StorageTier, amount: Long): Unit = {}

      override def semaphoreWaitTime: GpuMetric = null
    }

    val handle3 =
      catalog.makeNewHandle(bufferId, -1000, mySpillCallback)
    withResource(catalog.acquireBuffer(handle3)) { buff =>
      assertResult(mySpillCallback)(buff.getSpillCallback)
    }

    // removing handles brings back the prior inserted callback
    // low priority that is remaining
    handle3.close()
    withResource(catalog.acquireBuffer(handle2)) { buff =>
      assertResult(RapidsBuffer.defaultSpillCallback)(buff.getSpillCallback)
    }

    handle2.close()
    withResource(catalog.acquireBuffer(handle1)) { buff =>
      assertResult(null)(buff.getSpillCallback)
    }

    handle1.close()
  }

  test("buffer registering slower tier does not hide faster tier") {
    val catalog = new RapidsBufferCatalog
    val bufferId = MockBufferId(5)
    val buffer = mockBuffer(bufferId, tier = StorageTier.DEVICE)
    val handle = catalog.registerNewBuffer(buffer)
    val buffer2 = mockBuffer(bufferId, tier = StorageTier.HOST)
    catalog.registerBuffer(buffer2)
    val buffer3 = mockBuffer(bufferId, tier = StorageTier.DISK)
    catalog.registerBuffer(buffer3)
    val acquired = catalog.acquireBuffer(handle)
    assertResult(5)(acquired.id.tableId)
    assertResult(buffer)(acquired)

    // registering the handle acquires the buffer
    verify(buffer, times(2)).addReference()
  }

  test("acquire buffer") {
    val catalog = new RapidsBufferCatalog
    val bufferId = MockBufferId(5)
    val buffer = mockBuffer(bufferId)
    val handle = catalog.registerNewBuffer(buffer)
    val acquired = catalog.acquireBuffer(handle)
    assertResult(5)(acquired.id.tableId)
    assertResult(buffer)(acquired)

    // registering the handle acquires the buffer
    verify(buffer, times(2)).addReference()
  }

  test("acquire buffer retries automatically") {
    val catalog = new RapidsBufferCatalog
    val bufferId = MockBufferId(5)
    val buffer = mockBuffer(bufferId, acquireAttempts = 9)
    val handle = catalog.registerNewBuffer(buffer)
    val acquired = catalog.acquireBuffer(handle)
    assertResult(5)(acquired.id.tableId)
    assertResult(buffer)(acquired)

    // registering the handle acquires the buffer
    verify(buffer, times(10)).addReference()
  }

  test("acquire buffer at specific tier") {
    val catalog = new RapidsBufferCatalog
    val bufferId = MockBufferId(5)
    val buffer = mockBuffer(bufferId, tier = StorageTier.DEVICE)
    catalog.registerBuffer(buffer)
    val buffer2 = mockBuffer(bufferId, tier = StorageTier.HOST)
    catalog.registerBuffer(buffer2)
    val acquired = catalog.acquireBuffer(MockBufferId(5), StorageTier.HOST).get
    assertResult(5)(acquired.id.tableId)
    assertResult(buffer2)(acquired)
    verify(buffer2).addReference()
  }

  test("acquire buffer at nonexistent tier") {
    val catalog = new RapidsBufferCatalog
    val bufferId = MockBufferId(5)
    val buffer = mockBuffer(bufferId, tier = StorageTier.HOST)
    catalog.registerBuffer(buffer)
    assert(catalog.acquireBuffer(MockBufferId(5), StorageTier.DEVICE).isEmpty)
    assert(catalog.acquireBuffer(MockBufferId(5), StorageTier.DISK).isEmpty)
  }

  test("get buffer meta") {
    val catalog = new RapidsBufferCatalog
    val bufferId = MockBufferId(5)
    val expectedMeta = new TableMeta
    val buffer = mockBuffer(bufferId, tableMeta = expectedMeta)
    catalog.registerBuffer(buffer)
    val meta = catalog.getBufferMeta(bufferId)
    assertResult(expectedMeta)(meta)
  }

  test("buffer is spilled to slower tier only") {
    val catalog = new RapidsBufferCatalog
    val bufferId = MockBufferId(5)
    val buffer = mockBuffer(bufferId, tier = StorageTier.DEVICE)
    catalog.registerBuffer(buffer)
    val buffer2 = mockBuffer(bufferId, tier = StorageTier.HOST)
    catalog.registerBuffer(buffer2)
    val buffer3 = mockBuffer(bufferId, tier = StorageTier.DISK)
    catalog.registerBuffer(buffer3)
    assert(catalog.isBufferSpilled(bufferId, StorageTier.DEVICE))
    assert(catalog.isBufferSpilled(bufferId, StorageTier.HOST))
    assert(!catalog.isBufferSpilled(bufferId, StorageTier.DISK))
  }

  test("multiple calls to unspill return existing DEVICE buffer") {
    val deviceStore = spy(new RapidsDeviceMemoryStore)
    val mockStore = mock[RapidsBufferStore]
    val hostStore = new RapidsHostMemoryStore(10000, 1000)
    deviceStore.setSpillStore(hostStore)
    hostStore.setSpillStore(mockStore)
    val catalog = new RapidsBufferCatalog(deviceStore)
    val handle = withResource(DeviceMemoryBuffer.allocate(1024)) { buff =>
      val meta = MetaUtils.getTableMetaNoTable(buff)
      catalog.addBuffer(
        buff, meta, -1, RapidsBuffer.defaultSpillCallback)
    }
    withResource(handle) { _ =>
      catalog.synchronousSpill(deviceStore, 0)
      val acquiredHostBuffer = catalog.acquireBuffer(handle)
      withResource(acquiredHostBuffer) { _ =>
        assertResult(HOST)(acquiredHostBuffer.storageTier)
        val unspilled =
          catalog.unspillBufferToDeviceStore(
            acquiredHostBuffer,
            acquiredHostBuffer.getMemoryBuffer,
            Cuda.DEFAULT_STREAM)
        withResource(unspilled) { _ =>
          assertResult(DEVICE)(unspilled.storageTier)
        }
        val unspilledSame = catalog.unspillBufferToDeviceStore(
          acquiredHostBuffer,
          acquiredHostBuffer.getMemoryBuffer,
          Cuda.DEFAULT_STREAM)
        withResource(unspilledSame) { _ =>
          assertResult(unspilled)(unspilledSame)
        }
        // verify that we invoked the copy function exactly once
        verify(deviceStore, times(1)).copyBuffer(any(), any(), any())
      }
    }
  }

  test("remove buffer tier") {
    val catalog = new RapidsBufferCatalog
    val bufferId = MockBufferId(5)
    val buffer = mockBuffer(bufferId, tier = StorageTier.DEVICE)
    catalog.registerBuffer(buffer)
    val buffer2 = mockBuffer(bufferId, tier = StorageTier.HOST)
    catalog.registerBuffer(buffer2)
    val buffer3 = mockBuffer(bufferId, tier = StorageTier.DISK)
    catalog.registerBuffer(buffer3)
    catalog.removeBufferTier(bufferId, StorageTier.DEVICE)
    catalog.removeBufferTier(bufferId, StorageTier.DISK)
    assert(catalog.acquireBuffer(MockBufferId(5), StorageTier.DEVICE).isEmpty)
    assert(catalog.acquireBuffer(MockBufferId(5), StorageTier.HOST).isDefined)
    assert(catalog.acquireBuffer(MockBufferId(5), StorageTier.DISK).isEmpty)
  }

  test("remove nonexistent buffer tier") {
    val catalog = new RapidsBufferCatalog
    val bufferId = MockBufferId(5)
    val buffer = mockBuffer(bufferId, tier = StorageTier.DEVICE)
    catalog.registerBuffer(buffer)
    catalog.removeBufferTier(bufferId, StorageTier.HOST)
    catalog.removeBufferTier(bufferId, StorageTier.DISK)
    assert(catalog.acquireBuffer(MockBufferId(5), StorageTier.DEVICE).isDefined)
    assert(catalog.acquireBuffer(MockBufferId(5), StorageTier.HOST).isEmpty)
    assert(catalog.acquireBuffer(MockBufferId(5), StorageTier.DISK).isEmpty)
  }

  test("remove buffer releases buffer resources") {
    val catalog = new RapidsBufferCatalog
    val bufferId = MockBufferId(5)
    val buffer = mockBuffer(bufferId)
    catalog.registerBuffer(buffer)
    val handle = catalog.makeNewHandle(
      bufferId, -1, RapidsBuffer.defaultSpillCallback)
    handle.close()
    verify(buffer).free()
  }

  test("remove buffer releases buffer resources at all tiers") {
    val catalog = new RapidsBufferCatalog
    val bufferId = MockBufferId(5)
    val buffer = mockBuffer(bufferId, tier = StorageTier.DEVICE)
    catalog.registerBuffer(buffer)
    val handle = catalog.makeNewHandle(
      bufferId, -1, RapidsBuffer.defaultSpillCallback)

    // these next registrations don't get their own handle. This is an internal
    // operation from the store where it has spilled to host and disk the RapidsBuffer
    val buffer2 = mockBuffer(bufferId, tier = StorageTier.HOST)
    catalog.registerBuffer(buffer2)
    val buffer3 = mockBuffer(bufferId, tier = StorageTier.DISK)
    catalog.registerBuffer(buffer3)

    // removing the original handle removes all buffers from all tiers.
    handle.close()
    verify(buffer).free()
    verify(buffer2).free()
    verify(buffer3).free()
  }

  private def mockBuffer(
      bufferId: RapidsBufferId,
      tableMeta: TableMeta = null,
      tier: StorageTier = StorageTier.DEVICE,
      acquireAttempts: Int = 1,
      initialPriority: Long = -1): RapidsBuffer = {
    spy(new RapidsBuffer {
      var _acquireAttempts: Int = acquireAttempts
      var currentPriority: Long = initialPriority
      var currentCallback: SpillMetricsCallback = null
      override val id: RapidsBufferId = bufferId
      override val size: Long = 0
      override val meta: TableMeta = tableMeta
      override val storageTier: StorageTier = tier

      override def getMemoryBuffer: MemoryBuffer = null

      override def copyToMemoryBuffer(
          srcOffset: Long,
          dst: MemoryBuffer,
          dstOffset: Long,
          length: Long,
          stream: Cuda.Stream): Unit = {}

      override def getDeviceMemoryBuffer: DeviceMemoryBuffer = null

      override def addReference(): Boolean = {
        if (_acquireAttempts > 0) {
          _acquireAttempts -= 1
        }
        _acquireAttempts == 0
      }

      override def free(): Unit = {}

      override def getSpillPriority: Long = currentPriority

      override def getSpillCallback: SpillMetricsCallback = currentCallback

      override def setSpillPriority(priority: Long): Unit = {
        currentPriority = priority
      }

      override def setSpillCallback(spillCallback: SpillMetricsCallback): Unit = {
        currentCallback = spillCallback
      }

      override def close(): Unit = {}
    })
  }

  case class MockBufferId(override val tableId: Int) extends RapidsBufferId {
    override def getDiskPath(dbm: RapidsDiskBlockManager): File =
      throw new UnsupportedOperationException
  }
}
