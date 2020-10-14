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

import ai.rapids.cudf.{CudaUtil, DeviceMemoryBuffer, HostMemoryBuffer, MemoryBuffer}
import com.nvidia.spark.rapids.format.{BufferMeta, TableMeta}
import org.mockito.{ArgumentCaptor, ArgumentMatchers}
import org.mockito.ArgumentMatchers._
import org.mockito.Mockito._

class RapidsShuffleClientSuite extends RapidsShuffleTestHelper {

  def prepareBufferReceiveState(
      tableMeta: TableMeta,
      bounceBuffer: BounceBuffer): BufferReceiveState = {
    val ptr = PendingTransferRequest(client, tableMeta, 123L, mockHandler)
    spy(new BufferReceiveState(bounceBuffer, Seq(ptr)))
  }

  def prepareBufferReceiveState(
      tableMetas: Seq[TableMeta],
      bounceBuffer: BounceBuffer): BufferReceiveState = {

    var tag = 123
    val ptrs = tableMetas.map { tm =>
      val ptr = PendingTransferRequest(client, tm, tag, mockHandler)
      val targetAlt = mock[AddressLengthTag]
      tag = tag + 1
      ptr
    }

    spy(new BufferReceiveState(bounceBuffer, ptrs))
  }

  def verifyTableMeta(expected: TableMeta, actual: TableMeta): Unit = {
    assertResult(expected.rowCount())(actual.rowCount())
    assertResult(expected.columnMetasLength())(actual.columnMetasLength())
    verifyBufferMeta(expected.bufferMeta, actual.bufferMeta)
  }

  def verifyBufferMeta(expected: BufferMeta, actual: BufferMeta): Unit = {
    assertResult(expected.id)(actual.id)
    assertResult(expected.size)(actual.size)
    assertResult(expected.uncompressedSize)(actual.uncompressedSize)
    assertResult(expected.codecBufferDescrsLength)(actual.codecBufferDescrsLength)
    (0 until expected.codecBufferDescrsLength).foreach { i =>
      val expectedDescr = expected.codecBufferDescrs(i)
      val actualDescr = actual.codecBufferDescrs(i)
      assertResult(expectedDescr.codec)(actualDescr.codec)
      assertResult(expectedDescr.compressedOffset)(actualDescr.compressedOffset)
      assertResult(expectedDescr.compressedSize)(actualDescr.compressedSize)
      assertResult(expectedDescr.uncompressedOffset)(actualDescr.uncompressedOffset)
      assertResult(expectedDescr.uncompressedSize)(actualDescr.uncompressedSize)
    }
  }

  test("successful metadata fetch") {
    when(mockTransaction.getStatus).thenReturn(TransactionStatus.Success)
    val shuffleRequests = RapidsShuffleTestHelper.getShuffleBlocks
    val contigBuffSize = 100000
    val numBatches = 3
    val tableMetas =
      RapidsShuffleTestHelper.mockMetaResponse(mockTransport, contigBuffSize, numBatches)

    // initialize metadata fetch
    client.doFetch(shuffleRequests.map(_._1), mockHandler)

    // the connection saw one request (for metadata)
    assertResult(1)(mockConnection.requests.size)

    // upon a successful response, the `start()` method in the fetch handler
    // will be called with 3 expected batches
    verify(mockHandler, times(1)).start(ArgumentMatchers.eq(numBatches))

    // the transport will receive 3 pending requests (for buffers) for queuing
    val ac = ArgumentCaptor.forClass(classOf[Seq[PendingTransferRequest]])
    verify(mockTransport, times(1)).queuePending(ac.capture())
    val ptrs = ac.getValue.asInstanceOf[Seq[PendingTransferRequest]]
    assertResult(numBatches)(ptrs.size)

    // we check their metadata below
    (0 until numBatches).foreach { t =>
      val expected = tableMetas(t)
      val tm = ptrs(t).tableMeta
      verifyTableMeta(expected, tm)
    }
  }

  test("successful degenerate metadata fetch") {
    when(mockTransaction.getStatus).thenReturn(TransactionStatus.Success)
    val shuffleRequests = RapidsShuffleTestHelper.getShuffleBlocks
    val numRows = 100000
    val numBatches = 3

    RapidsShuffleTestHelper.mockDegenerateMetaResponse(mockTransport, numRows, numBatches)

    // initialize metadata fetch
    client.doFetch(shuffleRequests.map(_._1), mockHandler)

    // the connection saw one request (for metadata)
    assertResult(1)(mockConnection.requests.size)

    // upon a successful response, the `start()` method in the fetch handler
    // will be called with 3 expected batches
    verify(mockHandler, times(1)).start(ArgumentMatchers.eq(numBatches))

    // nothing gets queued to be received since it's just metadata
    verify(mockTransport, times(0)).queuePending(any())

    // ensure our handler (iterator) received 3 batches
    verify(mockHandler, times(numBatches)).batchReceived(any())
  }

  test("errored/cancelled metadata fetch") {
    Seq(TransactionStatus.Error, TransactionStatus.Cancelled).foreach { status =>
      when(mockTransaction.getStatus).thenReturn(status)
      when(mockTransaction.getErrorMessage).thenReturn(Some("Error/cancel occurred"))

      val shuffleRequests = RapidsShuffleTestHelper.getShuffleBlocks
      val contigBuffSize = 100000
      RapidsShuffleTestHelper.mockMetaResponse(
        mockTransport, contigBuffSize, 3)

      client.doFetch(shuffleRequests.map(_._1), mockHandler)

      assertResult(1)(mockConnection.requests.size)

      // upon an errored response, the start handler will not be called
      verify(mockHandler, times(0)).start(any())

      // but the error handler will
      verify(mockHandler, times(1)).transferError(anyString())

      // the transport will receive no pending requests (for buffers) for queuing
      verify(mockTransport, times(0)).queuePending(any())

      newMocks()
    }
  }

  test("successful buffer fetch") {
    when(mockTransaction.getStatus).thenReturn(TransactionStatus.Success)

    val numRows = 25001
    val tableMeta =
      RapidsShuffleTestHelper.prepareMetaTransferResponse(mockTransport, numRows)

    // 10000 in bytes ~ 2500 rows (minus validity/offset buffers) worth of contiguous
    // single column int table, so we need 10 buffer-lengths to receive all of 25000 rows,
    // the extra one adds 1 receive. Note that each receive is 2 buffers (except for the last one),
    // so that is 6 receives expected.
    val sizePerBuffer = 10000
    // 5 receives (each with 2 buffers) makes for 100000 bytes + 1 receive for the remaining byte
    val expectedReceives = 11

    val refHostBuffer = HostMemoryBuffer.allocate(100032)
    var count = 0
    (0 until refHostBuffer.getLength.toInt)
        .foreach { off =>
          refHostBuffer.setByte(off, count.toByte)
          count = count + 1
          if (count >= sizePerBuffer) {
            count = 0
          }
        }

    closeOnExcept(getBounceBuffer(sizePerBuffer)) { bounceBuffer =>
      val db = bounceBuffer.buffer.asInstanceOf[DeviceMemoryBuffer]
      db.copyFromHostBuffer(refHostBuffer.slice(0, sizePerBuffer))
      val brs = prepareBufferReceiveState(tableMeta, bounceBuffer)

      assert(brs.hasNext)

      // Kick off receives
      client.doIssueBufferReceives(brs)

      // If transactions are successful, we should have completed the receive
      assert(!brs.hasNext)

      // we would issue as many requests as required in order to get the full contiguous
      // buffer
      verify(mockConnection, times(expectedReceives))
        .receive(any[Seq[AddressLengthTag]](), any[TransactionCallback]())

      // the mock connection keeps track of every receive length
      val totalReceived = mockConnection.receiveLengths.sum
      val numBuffersUsed = mockConnection.receiveLengths.size

      assertResult(tableMeta.bufferMeta().size())(totalReceived)
      assertResult(11)(numBuffersUsed)

      // we would perform 1 request to issue a `TransferRequest`, so the server can start.
      verify(mockConnection, times(1)).request(any(), any(), any[TransactionCallback]())

      // we will hand off a `DeviceMemoryBuffer` to the catalog
      val dmbCaptor = ArgumentCaptor.forClass(classOf[DeviceMemoryBuffer])
      val tmCaptor = ArgumentCaptor.forClass(classOf[TableMeta])
      verify(client, times(1)).track(any[DeviceMemoryBuffer](), tmCaptor.capture())
      verifyTableMeta(tableMeta, tmCaptor.getValue.asInstanceOf[TableMeta])
      verify(mockStorage, times(1))
          .addBuffer(any(), dmbCaptor.capture(), any(), any())

      val receivedBuff = dmbCaptor.getValue.asInstanceOf[DeviceMemoryBuffer]
      assertResult(tableMeta.bufferMeta().size())(receivedBuff.getLength)

      var hostBuff = HostMemoryBuffer.allocate(receivedBuff.getLength)
      CudaUtil.copy(receivedBuff, 0, hostBuff, 0, receivedBuff.getLength)
      (0 until numRows).foreach { r =>
        assertResult(refHostBuffer.getByte(r))(hostBuff.getByte(r))
        println(s"passes for ${r}")
      }

      // after closing, we should have freed our bounce buffers.
      assertResult(true)(bounceBuffer.isClosed)
    }
  }

  test("successful buffer fetch multi-buffer") {
    when(mockTransaction.getStatus).thenReturn(TransactionStatus.Success)

    val numRows = 500
    val tableMetas =
      (0 until 5).map {
        _ => RapidsShuffleTestHelper.prepareMetaTransferResponse(mockTransport, numRows)
      }

    // 20000 in bytes ~ 5000 rows (minus validity/offset buffers) worth of contiguous
    // single column int table, so we can pack 5 device receives into a single bounce buffer
    val sizePerBuffer = 20000
    // 5 receives (each with 2 buffers) makes for 100000 bytes + 1 receive for the remaining byte
    val expectedReceives = 1

    closeOnExcept(getBounceBuffer(sizePerBuffer)) { bounceBuffer =>
      val brs = prepareBufferReceiveState(tableMetas, bounceBuffer)

      assert(brs.hasNext)

      // Kick off receives
      client.doIssueBufferReceives(brs)

      // If transactions are successful, we should have completed the receive
      assert(!brs.hasNext)

      // we would issue as many requests as required in order to get the full contiguous
      // buffer
      verify(mockConnection, times(expectedReceives))
          .receive(any[Seq[AddressLengthTag]](), any[TransactionCallback]())

      // the mock connection keeps track of every receive length
      val totalReceived = mockConnection.receiveLengths.sum
      val numBuffersUsed = mockConnection.receiveLengths.size

      val totalExpectedSize = tableMetas.map(tm => tm.bufferMeta().size()).sum
      assertResult(totalExpectedSize)(totalReceived)
      assertResult(1)(numBuffersUsed)

      // we would perform 1 request to issue a `TransferRequest`, so the server can start.
      verify(mockConnection, times(1)).request(any(), any(), any[TransactionCallback]())

      // we will hand off a `DeviceMemoryBuffer` to the catalog
      val dmbCaptor = ArgumentCaptor.forClass(classOf[DeviceMemoryBuffer])
      val tmCaptor = ArgumentCaptor.forClass(classOf[TableMeta])
      verify(client, times(5)).track(any[DeviceMemoryBuffer](), tmCaptor.capture())
      tableMetas.zipWithIndex.foreach { case (tm, ix) =>
        verifyTableMeta(tm, tmCaptor.getAllValues().get(ix).asInstanceOf[TableMeta])
      }

      verify(mockStorage, times(5))
          .addBuffer(any(), dmbCaptor.capture(), any(), any())

      assertResult(totalExpectedSize)(
        dmbCaptor.getAllValues().toArray().map(_.asInstanceOf[DeviceMemoryBuffer].getLength).sum)

      // after closing, we should have freed our bounce buffers.
      assertResult(true)(bounceBuffer.isClosed)
    }
  }

  test("successful buffer fetch multi-buffer, larger than a single bounce buffer") {
    when(mockTransaction.getStatus).thenReturn(TransactionStatus.Success)

    val numRows = 500
    val tableMetas =
      (0 until 20).map {
        _ => RapidsShuffleTestHelper.prepareMetaTransferResponse(mockTransport, numRows)
      }

    // 20000 in bytes ~ 5000 rows (minus validity/offset buffers) worth of contiguous
    // single column int table, so we can pack 5 device receives into a single bounce buffer
    // we have 20 bounce buffers, so we expect in this case 3 receives.
    val sizePerBuffer = 20000
    // 5 receives (each with 2 buffers) makes for 100000 bytes + 1 receive for the remaining byte
    val expectedReceives = 3

    closeOnExcept(getBounceBuffer(sizePerBuffer)) { bounceBuffer =>
      val brs = prepareBufferReceiveState(tableMetas, bounceBuffer)

      assert(brs.hasNext)

      // Kick off receives
      client.doIssueBufferReceives(brs)

      // If transactions are successful, we should have completed the receive
      assert(!brs.hasNext)

      // we would issue as many requests as required in order to get the full contiguous
      // buffer
      verify(mockConnection, times(expectedReceives))
          .receive(any[Seq[AddressLengthTag]](), any[TransactionCallback]())

      // the mock connection keeps track of every receive length
      val totalReceived = mockConnection.receiveLengths.sum
      val numBuffersUsed = mockConnection.receiveLengths.size

      val totalExpectedSize = tableMetas.map(tm => tm.bufferMeta().size()).sum
      assertResult(totalExpectedSize)(totalReceived)
      assertResult(3)(numBuffersUsed)

      // we would perform 1 request to issue a `TransferRequest`, so the server can start.
      verify(mockConnection, times(1)).request(any(), any(), any[TransactionCallback]())

      // we will hand off a `DeviceMemoryBuffer` to the catalog
      val dmbCaptor = ArgumentCaptor.forClass(classOf[DeviceMemoryBuffer])
      val tmCaptor = ArgumentCaptor.forClass(classOf[TableMeta])
      verify(client, times(20)).track(any[DeviceMemoryBuffer](), tmCaptor.capture())
      tableMetas.zipWithIndex.foreach { case (tm, ix) =>
        verifyTableMeta(tm, tmCaptor.getAllValues().get(ix).asInstanceOf[TableMeta])
      }

      verify(mockStorage, times(20))
          .addBuffer(any(), dmbCaptor.capture(), any(), any())

      assertResult(totalExpectedSize)(
        dmbCaptor.getAllValues().toArray().map(_.asInstanceOf[DeviceMemoryBuffer].getLength).sum)

      // after closing, we should have freed our bounce buffers.
      assertResult(true)(bounceBuffer.isClosed)
    }
  }

  test("errored/cancelled buffer fetch") {
    Seq(TransactionStatus.Error, TransactionStatus.Cancelled).foreach { status =>
      when(mockTransaction.getStatus).thenReturn(status)
      when(mockTransaction.getErrorMessage).thenReturn(Some("Error/cancel occurred"))

      val numRows = 100000
      val tableMeta =
        RapidsShuffleTestHelper.prepareMetaTransferResponse(mockTransport, numRows)

      // error condition, so it doesn't matter much what we set here, only the first
      // receive will happen
      val sizePerBuffer = numRows * 4 / 10
      closeOnExcept(getBounceBuffer(sizePerBuffer)) { bounceBuffer =>
        val brs = prepareBufferReceiveState(tableMeta, bounceBuffer)

        assert(brs.hasNext)

        // Kick off receives
        client.doIssueBufferReceives(brs)

        // Errored transaction. Therefore we should not be done
        assert(brs.hasNext)

        // We should have called `transferError` in the `RapidsShuffleFetchHandler`
        verify(mockHandler, times(1)).transferError(any())

        // there was 1 receive, and the chain stopped because it wasn't successful
        verify(mockConnection, times(1)).receive(any[Seq[AddressLengthTag]](), any())

        // we would have issued 1 request to issue a `TransferRequest` for the server to start
        verify(mockConnection, times(1)).request(any(), any(), any())

        // ensure we closed the BufferReceiveState => releasing the bounce buffers
        assertResult(true)(bounceBuffer.isClosed)
      }

      newMocks()
    }
  }

  test ("adding requests -- get a bounce buffer length") {

    withResource(getBounceBuffer(1000)) { buff =>

      val tableMeta =
        RapidsShuffleTestHelper.prepareMetaTransferResponse(mockTransport, 1024)

      val ptr = mock[PendingTransferRequest]
      when(ptr.tableMeta).thenReturn(tableMeta)
      when(ptr.getLength).thenReturn(tableMeta.bufferMeta().size())

      val br = new BufferReceiveState(buff, Seq(ptr))
      println(br.next())
      assert(br.hasNext)
      println(br.next())
      println(br.next())
      println(br.next())
      println(br.next())

      assert(!br.hasNext)
    }
  }

  test ("adding requests -- get a bounce buffer length -- requests span two bounce buffer") {
    withResource(getBounceBuffer(1000)) { buff =>

      val ptr = mock[PendingTransferRequest]
      val ptr2 = mock[PendingTransferRequest]

      val mockTable = RapidsShuffleTestHelper.mockTableMeta(40)
      val mockTable2 = RapidsShuffleTestHelper.mockTableMeta(200)

      when(ptr.getLength).thenReturn(mockTable.bufferMeta().size())
      when(ptr.tableMeta).thenReturn(mockTable)

      when(ptr2.getLength).thenReturn(mockTable2.bufferMeta().size())
      when(ptr2.tableMeta).thenReturn(mockTable2)

      val br = new BufferReceiveState(buff, ptr :: ptr2 :: Nil)

      val state = br.next()

      println(state)

      assert(br.hasNext)

      val state2 = br.next()

      println(state2)

      assert(!br.hasNext)
    }
  }

  test ("adding requests -- get a bounce buffer length -- requests span three bounce buffer " +
      "consume") {
    withResource(getBounceBuffer(1000)) { buff =>
      val ptr = mock[PendingTransferRequest]
      val ptr2 = mock[PendingTransferRequest]
      val ptr3 = mock[PendingTransferRequest]

      val mockTable = RapidsShuffleTestHelper.mockTableMeta(40)
      val mockTable2 = RapidsShuffleTestHelper.mockTableMeta(180)
      val mockTable3 = RapidsShuffleTestHelper.mockTableMeta(100)

      when(ptr.getLength).thenReturn(mockTable.bufferMeta().size())
      when(ptr.tableMeta).thenReturn(mockTable)

      when(ptr2.getLength).thenReturn(mockTable2.bufferMeta().size())
      when(ptr2.tableMeta).thenReturn(mockTable2)

      when(ptr3.getLength).thenReturn(mockTable3.bufferMeta().size())
      when(ptr3.tableMeta).thenReturn(mockTable3)

      val br = new BufferReceiveState(buff, ptr :: ptr2 :: ptr3 :: Nil)

      val state = br.next()
      br.consumeWindow()

      println(state)
      assert(br.hasNext)

      val state2 = br.next()
      br.consumeWindow()
      println(state2)

      assert(!br.hasNext)


    }
  }

  test ("adding requests -- get a bounce buffer length -- request larger than bb" +
      "consume") {

    withResource(getBounceBuffer(1000)) { buff =>
      val ptr = mock[PendingTransferRequest]
      val ptr2 = mock[PendingTransferRequest]

      val mockTable = RapidsShuffleTestHelper.mockTableMeta(300)
      val mockTable2 = RapidsShuffleTestHelper.mockTableMeta(300)

      when(ptr.getLength).thenReturn(mockTable.bufferMeta().size())
      when(ptr.tableMeta).thenReturn(mockTable)

      when(ptr2.getLength).thenReturn(mockTable2.bufferMeta().size())
      when(ptr2.tableMeta).thenReturn(mockTable2)

      val br = new BufferReceiveState(buff, ptr :: ptr2 :: Nil)

      val state = br.next()
      br.consumeWindow()

      println(state)
      assert(br.hasNext)

      val state2 = br.next()
      br.consumeWindow()

      println(state2)

      assert(br.hasNext)

      val state3 = br.next()
      br.consumeWindow()

      println(state3)
      // this should have beenc onsumed by nlow!!

      assert(!br.hasNext)
    }
  }

  test ("adding requests -- get a bounce buffer length -- request larger than bb " +
      "-- second one spans twice consume") {

    withResource(getBounceBuffer(1000)) { buff =>
      val ptr = mock[PendingTransferRequest]
      val ptr2 = mock[PendingTransferRequest]

      val mockTable = RapidsShuffleTestHelper.mockTableMeta(300)
      val mockTable2 = RapidsShuffleTestHelper.mockTableMeta(800)

      when(ptr.tag).thenReturn(1)
      when(ptr.getLength).thenReturn(mockTable.bufferMeta().size())
      when(ptr.tableMeta).thenReturn(mockTable)

      when(ptr2.tag).thenReturn(2)
      when(ptr2.getLength).thenReturn(mockTable2.bufferMeta().size())
      when(ptr2.tableMeta).thenReturn(mockTable2)

      val br = new BufferReceiveState(buff, ptr :: ptr2 :: Nil)

      val state = br.next()
      br.consumeWindow()

      println(state)
      assert(br.hasNext)

      val state2 = br.next()
      br.consumeWindow()

      println(state2)

      assert(br.hasNext)

      val state3 = br.next()
      br.consumeWindow()

      println(state3)

      assert(br.hasNext)

      val state4 = br.next()
      br.consumeWindow()

      println(state4)

      assert(br.hasNext)

      val state5 = br.next()
      br.consumeWindow()

      println(state5)

      assert(!br.hasNext)
    }
  }
}
