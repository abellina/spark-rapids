package org.apache.spark.shuffle

import java.nio.ByteBuffer

import ai.rapids.cudf.MemoryBuffer
import com.nvidia.spark.rapids.{RapidsBufferId, ShuffleBufferId, ShuffleMetadata}
import com.nvidia.spark.rapids.format.TableMeta
import org.openucx.jucx.UcxUtils

import org.apache.spark.shuffle.ucx.{Block, BlockId, MemoryBlock}
import org.apache.spark.storage.ShuffleBlockBatchId

case class RapidsBlockId(bufferId: RapidsBufferId) extends BlockId
case class RapidsMetaBlockId(shuffleBlockId: String) extends BlockId
class RapidsMetaResponse(val bb: ByteBuffer)
    extends MemoryBlock(UcxUtils.getAddress(bb), bb.capacity())

class RapidsShuffleMetaBlock(
    shuffleBlockId: ShuffleBlockBatchId,
    tableMetas: Seq[TableMeta]) extends Block {

  val rapidsMetaBlockId = RapidsMetaBlockId(shuffleBlockId.name)

  private val res: ByteBuffer =
    ShuffleMetadata.buildBlockMeta(tableMetas, 1024L * 1024L)

  override def getMemoryBlock: MemoryBlock = {
    MemoryBlock(UcxUtils.getAddress(res), res.remaining(), true)
  }

  def getBlockId: RapidsMetaBlockId = rapidsMetaBlockId
}

class RapidsShuffleBlock(bufferId: ShuffleBufferId,
    address: Long, length: Long, meta: TableMeta) extends Block {
  override def getMemoryBlock: MemoryBlock = {
    MemoryBlock(address, length, false)
  }

  val rapidsBlockId = RapidsBlockId(bufferId)

  def getBlockId: RapidsBlockId = rapidsBlockId
}
