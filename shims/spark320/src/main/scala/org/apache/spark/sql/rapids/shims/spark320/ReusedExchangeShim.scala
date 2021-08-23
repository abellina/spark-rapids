package org.apache.spark.sql.rapids.shims.spark320

import org.apache.spark.sql.catalyst.expressions.Expression
import org.apache.spark.sql.execution.exchange.ReusedExchangeExec

object ReusedExchangeShim {
  def updateAttr(
      expression: Expression,
      reusedExchange: ReusedExchangeExec): Expression = {
    reusedExchange.updateAttr(expression)
  }
}
