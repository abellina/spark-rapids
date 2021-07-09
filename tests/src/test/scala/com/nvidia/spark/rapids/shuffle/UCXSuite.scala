package com.nvidia.spark.rapids.shuffle.ucx

import com.nvidia.spark.rapids.RapidsConf
import com.nvidia.spark.rapids.shuffle.{RapidsShuffleTestHelper, RefCountedDirectByteBuffer}
import com.nvidia.spark.rapids.shuffle.ucx.{UCX, UCXActiveMessage, UCXAmCallback, UCXShuffleTransport}
import org.openucx.jucx.ucp.UcpRequest
import org.scalatest.FunSuite
import org.scalatest.mockito.MockitoSugar

class UCXSuite extends FunSuite with MockitoSugar {
  test("ucx active message registration") {
    val u = new UCX(
      mock[UCXShuffleTransport],
      RapidsShuffleTestHelper.makeMockBlockManager("1", "test"),
      mock[RapidsConf])

    u.registerResponseHandler(UCXActiveMessage(1,2), new UCXAmCallback {
      override def onError(am: UCXActiveMessage, error: UCXError): Unit = {}
      override def onMessageStarted(receiveAm: UcpRequest): Unit = {}
      override def onSuccess(am: UCXActiveMessage, buff: RefCountedDirectByteBuffer): Unit = {}
      override def onCancel(am: UCXActiveMessage): Unit = {}
      override def onHostMessageReceived(size: Long): RefCountedDirectByteBuffer = {null}
    })

    u.unregisterResponseHandler(UCXActiveMessage(1,2))
  }
}
