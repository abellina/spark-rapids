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

import com.nvidia.spark.rapids.{Arm, NoopMetric, SpillableColumnarBatchImpl}
import org.mockito.Mockito.when
import org.scalatest.FunSuite
import org.scalatest.mockito.MockitoSugar

import org.apache.spark.sql.types.{DataType, IntegerType}

class SpillableColumnarBatchSuite extends FunSuite with Arm with MockitoSugar {
  test("close updates catalog") {
    val mockBuffer = mock[RapidsBuffer]
    val mockId = TempSpillBufferId()
    when(mockBuffer.id).thenReturn(mockId)
    val catalog = new RapidsBufferCatalog(mock[RapidsDeviceMemoryStore])
    val oldBufferCount = catalog.numBuffers
    val handle = catalog.registerNewBuffer(
      mockBuffer, -1, RapidsBuffer.defaultSpillCallback)
    assertResult(oldBufferCount + 1)(catalog.numBuffers)
    val spillableBatch = new SpillableColumnarBatchImpl(
      handle,
      5,
      Array[DataType](IntegerType),
      NoopMetric)
    spillableBatch.close()
    assertResult(oldBufferCount)(catalog.numBuffers)
  }
}
