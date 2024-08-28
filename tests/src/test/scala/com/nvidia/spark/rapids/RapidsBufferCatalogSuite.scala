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

package com.nvidia.spark.rapids

import java.io.File

import ai.rapids.cudf.{Cuda, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer}
import com.nvidia.spark.rapids.Arm.withResource
import com.nvidia.spark.rapids.StorageTier.{DEVICE, DISK, HOST, StorageTier}
import com.nvidia.spark.rapids.format.TableMeta
import org.mockito.ArgumentMatchers.any
import org.mockito.Mockito._
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite
import org.scalatestplus.mockito.MockitoSugar

import org.apache.spark.sql.rapids.RapidsDiskBlockManager
import org.apache.spark.sql.types.DataType
import org.apache.spark.sql.vectorized.ColumnarBatch

class RapidsBufferCatalogSuite
  extends AnyFunSuite
    with MockitoSugar
    with BeforeAndAfterEach {

  test("lookup unknown buffer") {
    withResource(new RapidsBufferCatalog()) { catalog =>
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
      //TODO: assertThrows[NoSuchElementException](catalog.getBufferMeta(bufferId))
    }
  }

  test("buffer double register throws") {
    withResource(new RapidsBufferCatalog()) { catalog =>
      val bufferId = MockBufferId(5)
      val buffer = mockBuffer(bufferId, catalog = catalog)
      catalog.registerNewBuffer(buffer)
      val buffer2 = mockBuffer(bufferId, catalog = catalog)
      assertThrows[DuplicateBufferException](catalog.registerNewBuffer(buffer2))
    }
  }

  test("a second handle prevents buffer to be removed") {
    withResource(new RapidsDeviceMemoryStore) { devStore =>
      withResource(new RapidsBufferCatalog(devStore)) { catalog =>
        val bufferId = MockBufferId(5)
        val buffer = mockBuffer(bufferId, catalog = catalog)
        catalog.registerNewBuffer(buffer)
        val handle1 =
          catalog.makeNewHandle(bufferId, -1)
        val handle2 =
          catalog.makeNewHandle(bufferId, -1)

        handle1.close()

        // this does not throw
        catalog.acquireBuffer(handle2).close()
        // actually this doesn't throw either
        catalog.acquireBuffer(handle1).close()

        handle2.close()

        assertThrows[NoSuchElementException](catalog.acquireBuffer(handle1))
        assertThrows[NoSuchElementException](catalog.acquireBuffer(handle2))
      }
    }
  }

  test("spill priorities are updated as handles are registered and unregistered") {
    withResource(new RapidsDeviceMemoryStore) { devStore =>
      withResource(new RapidsBufferCatalog(devStore)) { catalog =>
        val bufferId = MockBufferId(5)
        val buffer = mockBuffer(bufferId, initialPriority = -1, catalog = catalog)
        catalog.registerNewBuffer(buffer)
        val handle1 =
          catalog.makeNewHandle(bufferId, -1)
        withResource(catalog.acquireBuffer(handle1)) { buff =>
          assertResult(-1)(buff.getSpillPriority)
        }
        val handle2 =
          catalog.makeNewHandle(bufferId, 0)
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
          catalog.makeNewHandle(bufferId, -1000)
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
    }
  }

  test("buffer registering slower tier does not hide faster tier") {
    withResource(new RapidsDeviceMemoryStore) { deviceStore =>
      val mockStore = mock[RapidsDiskStore]
      when(mockStore.tier).thenReturn(StorageTier.DISK)
      withResource(
        new RapidsHostMemoryStore(Some(10000))) { hostStore =>
        deviceStore.setSpillStore(hostStore)
        hostStore.setSpillStore(mockStore)
        withResource(new RapidsBufferCatalog(deviceStore, hostStore, mockStore)) { catalog =>
          val bufferId = MockBufferId(5)
          val buffer = mockBuffer(bufferId, tier = DEVICE, catalog = catalog)
          catalog.registerNewBuffer(buffer)
          val handle = catalog.makeNewHandle(bufferId, 0)
          buffer.copyTo(hostStore, Cuda.DEFAULT_STREAM) // now it's also on host
          buffer.copyTo(mockStore, Cuda.DEFAULT_STREAM) // now it's also on mockStore "disk"
          //val buffer2 = mockBuffer(bufferId, tier = HOST, catalog = catalog)
          //catalog.registerNewBuffer(buffer2)
          //val buffer3 = mockBuffer(bufferId, tier = DISK, catalog = catalog)
          //catalog.registerNewBuffer(buffer3)
          val acquired = catalog.acquireBuffer(handle)
          assertResult(5)(acquired.id.tableId)
          assertResult(buffer)(acquired.base)
          assertResult(StorageTier.DEVICE)(acquired.storageTier)

          verify(buffer.get(promote = false), times(1)).addReference()
        }
      }
    }
  }

  test("acquire buffer") {
    withResource(new RapidsDeviceMemoryStore) { devStore =>
      withResource(new RapidsBufferCatalog(devStore)) { catalog =>
        val bufferId = MockBufferId(5)
        val buffer = mockBuffer(bufferId, catalog = catalog)
        catalog.registerNewBuffer(buffer)
        val handle = catalog.makeNewHandle(bufferId, 0)
        val acquired = catalog.acquireBuffer(handle)
        assertResult(5)(acquired.id.tableId)
        assertResult(buffer)(acquired.base)
        verify(buffer.get(promote = false), times(1)).addReference()
      }
    }
  }

  test("acquire buffer retries automatically") {
    withResource(new RapidsDeviceMemoryStore) { devStore =>
      withResource(new RapidsBufferCatalog(devStore)) { catalog =>
        val bufferId = MockBufferId(5)
        val buffer = mockBuffer(bufferId, acquireAttempts = 9, catalog = catalog)
        catalog.registerNewBuffer(buffer)
        val handle = catalog.makeNewHandle(bufferId, 0)
        val acquired = catalog.acquireBuffer(handle)
        assertResult(5)(acquired.id.tableId)
        assertResult(buffer)(acquired.base)
        verify(buffer.get(promote = false), times(9)).addReference()
      }
    }
  }

  test("acquire buffer at specific tier") {
    val hostStore = mock[RapidsHostMemoryStore]
    when(hostStore.tier).thenReturn(StorageTier.HOST)

    withResource(new RapidsDeviceMemoryStore()) { devStore =>
      withResource(new RapidsBufferCatalog(
        deviceStorage = devStore,
        hostStorage = hostStore)) { catalog =>
        val bufferId = MockBufferId(5)
        val buffer = mockBuffer(bufferId, tier = DEVICE, catalog = catalog)
        catalog.registerNewBuffer(buffer)
        withResource(catalog.makeNewHandle(bufferId, 0)) { handle =>
          val buffer2 = buffer.copyTo(hostStore, Cuda.DEFAULT_STREAM)
          withResource(catalog.acquireBuffer(handle, HOST)) { acquired =>
            assertResult(5)(acquired.get.id.tableId)
            assertResult(buffer2.get)(acquired.get)
          }
        }
        //verify(buffer2.get.get(promote = false)).addReference()
      }
    }
  }

  test("acquire buffer at nonexistent tier") {
    val hostStore = mock[RapidsHostMemoryStore]
    when(hostStore.tier).thenReturn(StorageTier.HOST)
    withResource(new RapidsDeviceMemoryStore()) { devStore =>
      withResource(new RapidsBufferCatalog(
        deviceStorage = devStore,
        hostStorage = hostStore)) { catalog =>
        val bufferId = MockBufferId(5)
        val buffer = mockBuffer(bufferId, tier = HOST, catalog = catalog)
        catalog.registerNewBuffer(buffer)
        val handle = catalog.makeNewHandle(bufferId, 0)
        assert(catalog.acquireBuffer(handle, DEVICE).isEmpty)
        assert(catalog.acquireBuffer(handle, DISK).isEmpty)
      }
    }
  }

  test("get buffer meta") {
    withResource(new RapidsBufferCatalog()) { catalog =>
      val bufferId = MockBufferId(5)
      val expectedMeta = new TableMeta
      val buffer = mockBuffer(bufferId,
        tier = DEVICE, catalog = catalog, tableMeta = expectedMeta)
      catalog.registerNewBuffer(buffer)
      withResource(catalog.makeNewHandle(bufferId, 0L)) { handle =>
        withResource(catalog.acquireBuffer(handle)) {
          case bwm: RapidsBufferWithMeta =>
            assertResult(expectedMeta)(bwm.meta)
        }
      }
    }
  }

  test("buffer is spilled to slower tier only") {
    val mockStore = mock[RapidsDiskStore]
    when(mockStore.tier).thenReturn(StorageTier.DISK)
    withResource(new RapidsDeviceMemoryStore()) { devStore =>
      withResource(new RapidsHostMemoryStore(None)) { hostStore =>
        devStore.setSpillStore(hostStore)
        hostStore.setSpillStore(mockStore)

        withResource(new RapidsBufferCatalog(
            deviceStorage = devStore,
            hostStorage = hostStore,
            diskStorage = mockStore)) { catalog =>
          val bufferId = MockBufferId(5)
          val buffer = mockBuffer(bufferId, tier = DEVICE, catalog = catalog)
          buffer.spill(devStore, hostStore, Cuda.DEFAULT_STREAM)
          buffer.spill(hostStore, mockStore, Cuda.DEFAULT_STREAM)
          assertResult(None)(buffer.get(StorageTier.DEVICE))
          assertResult(None)(buffer.get(StorageTier.HOST))
          assertResult(true)(buffer.get(StorageTier.DISK).isDefined)
        }
      }
    }
  }

  test("multiple calls to unspill return existing DEVICE buffer") {
    withResource(spy(new RapidsDeviceMemoryStore)) { devStore =>
      val mockStore = mock[RapidsDiskStore]
      withResource(
        new RapidsHostMemoryStore(Some(10000))) { hostStore =>
        devStore.setSpillStore(hostStore)
        hostStore.setSpillStore(mockStore)
        withResource(new RapidsBufferCatalog(
          deviceStorage= devStore,
          hostStorage = hostStore,
          diskStorage = mockStore)) { catalog =>
          val handle = withResource(DeviceMemoryBuffer.allocate(1024)) { buff =>
            val meta = MetaUtils.getTableMetaNoTable(buff.getLength)
            catalog.addBufferWithMeta(
              buff, meta, -1)
          }
          var rmb: RapidsMemoryBuffer = null
          withResource(handle) { _ =>
            catalog.synchronousSpill(devStore, 0)
            val acquiredHostBuffer = catalog.acquireBuffer(handle)
            withResource(acquiredHostBuffer) { _ =>
              assertResult(HOST)(acquiredHostBuffer.storageTier)
              rmb = acquiredHostBuffer.base
              val unspilled = rmb.copyTo(devStore, Cuda.DEFAULT_STREAM).get
              assertResult(DEVICE)(unspilled.storageTier)
              val unspilledSame = rmb.copyTo(devStore, Cuda.DEFAULT_STREAM)
              assertResult(None)(unspilledSame) // already unspilled
              assertResult(unspilled)(rmb.get(StorageTier.DEVICE).get)
            }
          }
          assertResult(None)(rmb.get(StorageTier.DEVICE))
          assertResult(None)(rmb.get(StorageTier.HOST))
          assertResult(None)(rmb.get(StorageTier.DISK))
        }
      }
    }
  }

  test("remove buffer tier") {
    val mockStore = mock[RapidsDiskStore]
    when(mockStore.tier).thenReturn(StorageTier.DISK)
    withResource(new RapidsDeviceMemoryStore()) { devStore =>
      withResource(new RapidsHostMemoryStore(None)) { hostStore =>
        devStore.setSpillStore(hostStore)
        hostStore.setSpillStore(mockStore)

        withResource(new RapidsBufferCatalog(
          deviceStorage = devStore,
          hostStorage = hostStore,
          diskStorage = mockStore)) { catalog =>
          val bufferId = MockBufferId(5)
          val buffer = mockBuffer(bufferId, tier = DEVICE, catalog = catalog)
          buffer.copyTo(hostStore, Cuda.DEFAULT_STREAM)
          buffer.copyTo(mockStore, Cuda.DEFAULT_STREAM)
          buffer.free(StorageTier.DEVICE)
          buffer.free(StorageTier.DISK)
          assert(buffer.get(StorageTier.DEVICE).isEmpty)
          assert(buffer.get(StorageTier.HOST).isDefined)
          assert(buffer.get(StorageTier.DISK).isEmpty)
        }
      }
    }
  }

  test("remove nonexistent buffer tier") {
    val mockStore = mock[RapidsDiskStore]
    when(mockStore.tier).thenReturn(StorageTier.DISK)
    withResource(new RapidsDeviceMemoryStore()) { devStore =>
      withResource(new RapidsHostMemoryStore(None)) { hostStore =>
        devStore.setSpillStore(hostStore)
        hostStore.setSpillStore(mockStore)

        withResource(new RapidsBufferCatalog(
          deviceStorage = devStore,
          hostStorage = hostStore,
          diskStorage = mockStore)) { catalog =>
          val bufferId = MockBufferId(5)
          val buffer = mockBuffer(bufferId, tier = DEVICE, catalog = catalog)
          buffer.free(StorageTier.HOST)
          buffer.free(StorageTier.DISK)
          assert(buffer.get(StorageTier.DEVICE).isDefined)
          assert(buffer.get(StorageTier.HOST).isEmpty)
          assert(buffer.get(StorageTier.DISK).isEmpty)
        }
      }
    }
  }

  test("remove buffer releases buffer resources") {
    withResource(new RapidsDeviceMemoryStore) { devStore =>
      withResource(new RapidsBufferCatalog(devStore)) { catalog =>
        val bufferId = MockBufferId(5)
        val buffer = mockBuffer(bufferId, catalog = catalog)
        catalog.registerNewBuffer(buffer)
        val handle = catalog.makeNewHandle(bufferId, -1)
        handle.close()
        assertThrows[IllegalStateException] {
          buffer.get(promote = false)
        }
      }
    }
  }

  test("remove buffer releases buffer resources at all tiers") {
    withResource(new RapidsDeviceMemoryStore) { deviceStore =>
      val mockStore = mock[RapidsDiskStore]
      when(mockStore.tier).thenReturn(StorageTier.DISK)
      withResource(new RapidsHostMemoryStore(Some(1000))) { hostStore =>
        deviceStore.setSpillStore(hostStore)
        hostStore.setSpillStore(mockStore)
        withResource(new RapidsBufferCatalog(deviceStore, hostStore, mockStore)) { catalog =>
          val bufferId = MockBufferId(5)
          val buffer = mockBuffer(bufferId, tier = DEVICE, catalog = catalog)
          catalog.registerNewBuffer(buffer)
          val handle = catalog.makeNewHandle(bufferId, -1)

          // these next registrations don't get their own handle. This is an internal
          // operation from the store where it has spilled to host and disk the RapidsBuffer
          buffer.copyTo(hostStore, Cuda.DEFAULT_STREAM)
          buffer.copyTo(mockStore, Cuda.DEFAULT_STREAM)

          // removing the original handle removes all buffers from all tiers.
          handle.close()
          assertThrows[IllegalStateException] {
            buffer.get(promote = false)
          }
          // TODO: verify(buffer).free()
          // TODO: verify(buffer2).free()
          // TODO: verify(buffer3).free()
        }
      }
    }
  }

  private def mockBuffer(
      bufferId: RapidsBufferId,
      tableMeta: TableMeta = null,
      tier: StorageTier = StorageTier.DEVICE,
      acquireAttempts: Int = 1,
      initialPriority: Long = -1,
      catalog: RapidsBufferCatalog): RapidsMemoryBuffer = {
    val rmb = new RapidsMemoryBuffer(bufferId)
    def makeRapidsBufferWithMeta(thisTier: StorageTier): RapidsBuffer = {
      new MockBuffer(bufferId, thisTier,
        tableMeta, initialPriority, catalog, acquireAttempts, rmb)
    }
    val bufferBase = spy(makeRapidsBufferWithMeta(tier))
    rmb.initialize(bufferBase, tier)
    rmb
  }
}

case class MockBufferId(override val tableId: Int) extends RapidsBufferId {
  override def getDiskPath(dbm: RapidsDiskBlockManager): File =
    throw new UnsupportedOperationException
}

class MockBuffer(
    bufferId: RapidsBufferId,
    thisTier: StorageTier,
    tableMeta: TableMeta,
    initialPriority: Long,
    catalog: RapidsBufferCatalog,
    acquireAttempts: Int,
    rmb: RapidsMemoryBuffer)
    extends RapidsBuffer with RapidsBufferWithMeta {
  private var spillPriority = initialPriority
  var released: Boolean = false
  var _acquireAttempts: Int = acquireAttempts

  override def addReference(): Boolean = {
    if (_acquireAttempts > 0) {
      _acquireAttempts -= 1
    }
    _acquireAttempts == 0
  }
  override val base: RapidsMemoryBuffer = rmb
  override def getCopyIterator(stream: Cuda.Stream): RapidsBufferCopyIterator = null
  override def copyTo(store: RapidsBufferStore, stream: Cuda.Stream): RapidsBuffer = {
    new MockBuffer(bufferId, store.tier, tableMeta, initialPriority, catalog,
      acquireAttempts, rmb)
  }
  override val id: RapidsBufferId = bufferId
  override val memoryUsedBytes: Long = 0
  override def meta: TableMeta = tableMeta
  override val storageTier: StorageTier = thisTier
  override def getColumnarBatch(sparkTypes: Array[DataType], stream: Cuda.Stream): ColumnarBatch = null
  override def getMemoryBuffer(stream: Cuda.Stream): MemoryBuffer = null
  override def close(): Unit = {}

  override def free(): Unit = {}

  override def setSpillPriority(newPriority: Long): Unit = {
    spillPriority = newPriority
  }
  override def getSpillPriority: Long = spillPriority
}
