/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

package com.nvidia.spark.rapids.shuffle

import ai.rapids.cudf.DeviceMemoryBuffer
import com.nvidia.spark.rapids.RapidsBuffer
import com.nvidia.spark.rapids.format.TableMeta
import org.mockito.Mockito._
import org.mockito.invocation.InvocationOnMock
import org.mockito.stubbing.Answer

import org.apache.spark.storage.ShuffleBlockBatchId

class RapidsShuffleServerSuite extends RapidsShuffleTestHelper {

  test("all tables fit") {
    val mockTransferRequest = RapidsShuffleTestHelper.prepareMetaTransferRequest(10, 1000)

    withResource(getSendBounceBuffer(10000)) { bounceBuffer =>
      withResource((0 until 10).map(_ => DeviceMemoryBuffer.allocate(1000))) { deviceBuffers =>
        val mockBuffers = deviceBuffers.map { deviceBuffer =>
          deviceBuffer.incRefCount()
          val mockBuffer = mock[RapidsBuffer]
          val mockMeta = RapidsShuffleTestHelper.mockTableMeta(100000)
          when(mockBuffer.getMemoryBuffer).thenReturn(deviceBuffer)
          when(mockBuffer.size).thenReturn(1000)
          when(mockBuffer.meta).thenReturn(mockMeta)
          println(deviceBuffer)
          mockBuffer
        }

        val handler = new RapidsShuffleRequestHandler {
          override def getShuffleBufferMetas(
              shuffleBlockBatchId: ShuffleBlockBatchId): Seq[TableMeta] = {
            throw new NotImplementedError("getShuffleBufferMetas")
          }

          override def acquireShuffleBuffer(tableId: Int): RapidsBuffer = {
            println(s"acquiring!! ${tableId}")
            mockBuffers(tableId)
          }
        }

        val bss = new BufferSendState(mockTransferRequest, bounceBuffer, handler)
        assert(bss.hasNext)
        val buff = bss.next()
        print (buff)
      }
    }

    newMocks()
  }

  test("doesn't fit in one buffer length, need 2") {
    val mockTransferRequest = RapidsShuffleTestHelper.prepareMetaTransferRequest(20, 1000)

    withResource(getSendBounceBuffer(10000)) { bounceBuffer =>
      withResource((0 until 20).map(_ => DeviceMemoryBuffer.allocate(1000))) { deviceBuffers =>
        val mockBuffers = deviceBuffers.map { deviceBuffer =>
          deviceBuffer.incRefCount()
          val mockMeta = RapidsShuffleTestHelper.mockTableMeta(100000)
          val mockBuffer = mock[RapidsBuffer]
          when(mockBuffer.getMemoryBuffer).thenReturn(deviceBuffer)
          when(mockBuffer.size).thenReturn(1000)
          when(mockBuffer.meta).thenReturn(mockMeta)
          mockBuffer
        }

        val handler = new RapidsShuffleRequestHandler {
          override def getShuffleBufferMetas(
              shuffleBlockBatchId: ShuffleBlockBatchId): Seq[TableMeta] = {
            throw new NotImplementedError("getShuffleBufferMetas")
          }

          override def acquireShuffleBuffer(tableId: Int): RapidsBuffer = {
            println(s"acquiring!! ${tableId}")
            mockBuffers(tableId)
          }
        }

        val bss = new BufferSendState(mockTransferRequest, bounceBuffer, handler)

        var buffs = bss.next()
        assert(bss.hasNext)

        buffs = bss.next()
        assert(!bss.hasNext)
      }
    }
  }


  import scala.language.implicitConversions

  implicit def toAnswerWithArgs[T](f: InvocationOnMock => T) = new Answer[T] {
    override def answer(i: InvocationOnMock): T = f(i)
  }

  test("make a buffer send state - doesn't fit - large buffers") {
    val mockTransferRequest = RapidsShuffleTestHelper.prepareMetaTransferRequest(20, 10000)

    withResource(getSendBounceBuffer(10000)) { bounceBuffer =>
      withResource((0 until 20).map(_ => DeviceMemoryBuffer.allocate(123000))) { deviceBuffers =>
        deviceBuffers.foreach(_.incRefCount())

        val handler = new RapidsShuffleRequestHandler {
          override def getShuffleBufferMetas(
              shuffleBlockBatchId: ShuffleBlockBatchId): Seq[TableMeta] = {
            throw new NotImplementedError("getShuffleBufferMetas")
          }

          override def acquireShuffleBuffer(tableId: Int): RapidsBuffer = {
            val mockBuffer = mock[RapidsBuffer]
            val mockMeta = RapidsShuffleTestHelper.mockTableMeta(100000)

            when(mockBuffer.getMemoryBuffer).thenAnswer((i) =>{
              val buff = deviceBuffers(tableId)
              buff.incRefCount()
              buff
            })

            when(mockBuffer.size).thenReturn(deviceBuffers(tableId).getLength)
            when(mockBuffer.meta).thenReturn(mockMeta)
            mockBuffer
          }
        }

        val bss = new BufferSendState(mockTransferRequest, bounceBuffer, handler)
        (0 until 246).foreach(i => {
          println(s"at ${i} ${bss.hasNext}")
          if (!bss.hasNext) {
            println ("DONE!!!")
          } else {
            val buffs = bss.next()
            if (bss.hasNext) {
              println(bss)
              println(buffs)
            }
          }
        })

        assert(!bss.hasNext)
      }
    }
  }

  test("make a buffer send state - doesn't fit - 65 MB buffer") {
    val mockTransferRequest =
      RapidsShuffleTestHelper.prepareMetaTransferRequest(2, 10000)

    withResource(getSendBounceBuffer(4 * 1024 * 1024)) { bounceBuffer =>
      withResource(
          Seq(66264512, 65859584)
              .map(sz => DeviceMemoryBuffer.allocate(sz))) { deviceBuffers =>
        deviceBuffers.foreach(_.incRefCount())
        val mockBuffers = deviceBuffers.map { deviceBuffer =>
          val mockBuffer = mock[RapidsBuffer]
          val mockMeta = RapidsShuffleTestHelper.mockTableMeta(100000)
          when(mockBuffer.getMemoryBuffer).thenReturn(deviceBuffer)
          when(mockBuffer.size).thenReturn(deviceBuffer.getLength)
          when(mockBuffer.meta).thenReturn(mockMeta)
          mockBuffer
        }

        val handler = new RapidsShuffleRequestHandler {
          override def getShuffleBufferMetas(
              shuffleBlockBatchId: ShuffleBlockBatchId): Seq[TableMeta] = {
            throw new NotImplementedError("getShuffleBufferMetas")
          }

          override def acquireShuffleBuffer(tableId: Int): RapidsBuffer = {
            val mockBuffer = mock[RapidsBuffer]
            val mockMeta = RapidsShuffleTestHelper.mockTableMeta(100000)

            when(mockBuffer.getMemoryBuffer).thenAnswer((i) =>{
              val buff = deviceBuffers(tableId)
              buff.incRefCount()
              buff
            })

            when(mockBuffer.size).thenReturn(deviceBuffers(tableId).getLength)
            when(mockBuffer.meta).thenReturn(mockMeta)
            mockBuffer
          }
        }

        val bss = new BufferSendState(mockTransferRequest, bounceBuffer, handler)
        (0 until 32).foreach(i => {
          println(s"at ${i} ${bss.hasNext}")
          if (!bss.hasNext) {
            println ("DONE!!!")
          } else {
            val buffs = bss.next()
            if (bss.hasNext) {
              println(bss)
              println(buffs)
            }
          }
        })

        assert(!bss.hasNext)
      }
    }
  }
}
