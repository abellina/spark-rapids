package org.apache.spark.shuffle

import java.nio.ByteBuffer

import com.nvidia.spark.rapids.format.TableMeta
import com.nvidia.spark.rapids.{RapidsBufferId, ShuffleBufferId, ShuffleMetadata}
import org.openucx.jucx.UcxUtils
import org.apache.spark.shuffle.ucx.{Block, BlockId, MemoryBlock}
import org.apache.spark.storage.ShuffleBlockBatchId

case class RapidsBlockId(bufferId: RapidsBufferId) extends BlockId
case class RapidsMetaBlockId(shuffleBlockId: String) extends BlockId
class RapidsMetaResponse(val bb: ByteBuffer)
    extends MemoryBlock(UcxUtils.getAddress(bb), bb.capacity())

class RapidsShuffleMetaBlock(
    shuffleBlockId: ShuffleBlockBatchId,
    tableMetas: Seq[TableMeta],
    maxMetaSize: Long) extends Block {

  val rapidsMetaBlockId = RapidsMetaBlockId(shuffleBlockId.name)

  private val res: ByteBuffer =
    ShuffleMetadata.buildBlockMeta(tableMetas, maxMetaSize)

  override def getMemoryBlock: MemoryBlock = {
    MemoryBlock(UcxUtils.getAddress(res), res.remaining(), isHostMemory = true)
  }

  def getBlockId: RapidsMetaBlockId = rapidsMetaBlockId
}

class RapidsShuffleBlock(bufferId: ShuffleBufferId,
    address: Long, length: Long, meta: TableMeta) extends Block {
  override def getMemoryBlock: MemoryBlock = {
    MemoryBlock(address, length, isHostMemory = false)
  }

  val rapidsBlockId = RapidsBlockId(bufferId)

  def getBlockId: RapidsBlockId = rapidsBlockId
}
