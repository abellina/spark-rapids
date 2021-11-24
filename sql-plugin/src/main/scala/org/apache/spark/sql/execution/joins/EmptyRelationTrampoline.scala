package org.apache.spark.sql.execution.joins

object EmptyRelationTrampoline {
  def makeEmptyHashedRelation(): HashedRelation = {
    val emptyIter = Iterator.empty
    HashedRelation.apply(emptyIter, Seq.empty, 0)
  }
}
