/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
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

import java.io.{InputStream, OutputStream}

import org.apache.spark.SparkConf
import org.apache.spark.io.CompressionCodec
import org.apache.spark.sql.rapids.TempSpillBufferId
import org.apache.spark.sql.rapids.execution.TrampolineUtil


/**
 * It's a wrapper of Spark's SerializerManager, which supports compression and encryption
 * on data streams.
 * For compression, it's turned on/off via seperated Rapids configurations and the underlying
 * compression codec uses existing Spark's.
 * For encryption, it's controlled by Spark's configuration to turn on/off.
 * @param conf
 */
class RapidsSerializerManager (conf: SparkConf) {
  private lazy val compressSpill = TrampolineUtil.isCompressSpill(conf)

  private lazy val serializerManager = if (conf
    .getBoolean(RapidsConf.TEST_IO_ENCRYPTION.key,false)) {
    TrampolineUtil.createSerializerManager(conf)
  } else {
    TrampolineUtil.getSerializerManager
  }

  private lazy val compressionCodec: CompressionCodec = TrampolineUtil.createCodec(conf)

  def wrapStream(bufferId: RapidsBufferId, s: OutputStream): OutputStream = {
    wrapForCompression(bufferId, wrapForEncryption(s))
  }

  def wrapStream(bufferId: RapidsBufferId, s: InputStream): InputStream = {
    wrapForCompression(bufferId, wrapForEncryption(s))
  }

  def wrapForCompression(bufferId: RapidsBufferId, s: InputStream): InputStream = {
    if (shouldCompress(bufferId)) compressionCodec.compressedInputStream(s) else s
  }

  def wrapForCompression(bufferId: RapidsBufferId, s: OutputStream): OutputStream = {
    if (shouldCompress(bufferId)) compressionCodec.compressedOutputStream(s) else s
  }

  def wrapForEncryption(s: InputStream): InputStream = {
    if (serializerManager != null) serializerManager.wrapForEncryption(s) else s
  }

  def wrapForEncryption(s: OutputStream): OutputStream = {
    if (serializerManager != null) serializerManager.wrapForEncryption(s) else s
  }

  private def shouldCompress(bufferId: RapidsBufferId): Boolean = {
    bufferId match {
      case _: TempSpillBufferId => compressSpill
      case _: ShuffleBufferId | _: ShuffleReceivedBufferId => false
      case _ => false
    }
  }
}
