/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
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

import java.util.concurrent.{Executors, ScheduledExecutorService, TimeUnit}

import scala.collection.mutable.ArrayBuffer

import com.google.common.util.concurrent.ThreadFactoryBuilder
import java.util
import org.apache.commons.lang3.mutable.MutableLong

import org.apache.spark.api.plugin.PluginContext
import org.apache.spark.internal.Logging
import org.apache.spark.sql.rapids.{GpuShuffleEnv, RapidsShuffleInternalManagerBase}
import org.apache.spark.storage.BlockManagerId


/**
 * This is the first message sent from the executor to the driver.
 * @param id `BlockManagerId` for the executor
 */
case class RapidsExecutorStartupMsg(id: BlockManagerId)

/**
 * Executor heartbeat message.
 * This gives the driver an opportunity to respond with `RapidsExecutorUpdateMsg`
 */
case class RapidsExecutorHeartbeatMsg(id: BlockManagerId)

/**
 * Driver response to an startup or heartbeat message, with new (to the peer) executors
 * from the last heartbeat.
 */
case class RapidsExecutorUpdateMsg(ids: Array[BlockManagerId])

class RapidsShuffleHeartbeatManager(heartbeatIntervalMs: Long) extends Logging {
  private case class ExecutorRegistration(id: BlockManagerId, registrationTime: Long)

  private case class LastExecutorHeartbeat(id: BlockManagerId) {
    var lastHeartbeat: Long = System.currentTimeMillis()
  }

  // Executors ordered from most recently registered to least-recently registered
  private[this] val executors = new ArrayBuffer[ExecutorRegistration]()

  // A mapping of executor IDs to registration index (ordered by arrival)
  private[this] val lastRegistrationSeen = new util.HashMap[BlockManagerId, MutableLong]

  // Keep a min-heap with the executor using the last heartbeat received time
  private[this] val lastExecutorHeartbeats =
    new HashedPriorityQueue[LastExecutorHeartbeat](
      (e, e1) => java.lang.Long.compare(e.lastHeartbeat, e1.lastHeartbeat))

  /**
   * Called by the driver plugin to handle a new registration from an executor.
   * @param id `BlockManagerId` for the peer
   * @return `RapidsExecutorUpdateMsg` with all known executors.
   */
  def registerExecutor(id: BlockManagerId): RapidsExecutorUpdateMsg = synchronized {
    logDebug(s"Registration from RAPIDS executor at $id")
    require(!lastRegistrationSeen.containsKey(id), s"Executor $id already registered")

    lastExecutorHeartbeats.offer(LastExecutorHeartbeat(id))

    checkStaleExecutors()

    val allExecutors = executors.map(e => e.id).toArray

    val registration = ExecutorRegistration(id, System.nanoTime)
    executors.append(registration)

    lastRegistrationSeen.put(id, new MutableLong(registration.registrationTime))
    RapidsExecutorUpdateMsg(allExecutors)
  }

  private def checkStaleExecutors(): Unit = {
    // if we haven't seen the executor in 2 heartbeat intervals
    val heartbeatTimeout = System.currentTimeMillis() - (heartbeatIntervalMs * 2)
    if (lastExecutorHeartbeats.peek().lastHeartbeat < heartbeatTimeout) {
      // found at least 1 executor that has been inactive for twice the heartbeat interval
      // lets remove it from our list, and iterate to find others that may also be stale
      val toRemove = new ArrayBuffer[LastExecutorHeartbeat]()
      val lastHbsIter = lastExecutorHeartbeats.iterator()
      var validExecutor = false
      while (lastHbsIter.hasNext || !validExecutor) {
        val execHb = lastHbsIter.next()
        validExecutor = execHb.lastHeartbeat >= heartbeatTimeout
        if (!validExecutor) {
          toRemove.append(execHb)
        }
        // else, the time is within the valid range, so we can stop
      }
      toRemove.foreach(toRemoveHb => {
        lastExecutorHeartbeats.remove(toRemoveHb)
      })
    }
  }

  /**
   * Called by the driver plugin to handle an executor heartbeat.
   * @param id `BlockManagerId` for the peer
   * @return `RapidsExecutorUpdateMsg` with new executors, since the last heartbeat was received.
   */
  def executorHeartbeat(id: BlockManagerId): RapidsExecutorUpdateMsg = synchronized {
    val lastRegistration = lastRegistrationSeen.get(id)
    if (lastRegistration == null) {
      throw new IllegalStateException(s"Heartbeat from unknown executor $id")
    }

    lastExecutorHeartbeats.offer(LastExecutorHeartbeat(id))

    checkStaleExecutors()

    val newExecutors = new ArrayBuffer[BlockManagerId]
    val iter = executors.zipWithIndex.reverseIterator
    var done = false
    while (iter.hasNext && !done) {
      val (entry, index) = iter.next()

      if (index > lastRegistration.getValue) {
        if (entry.id != id) {
          newExecutors += entry.id
        }
      } else {
        // We are iterating backwards and have found the last index previously inspected
        // for this peer. We can stop since all peers below this have been sent to this peer.
        done = true
      }
    }

    lastRegistration.setValue(executors.size - 1)
    RapidsExecutorUpdateMsg(newExecutors.toArray)
  }
}

trait RapidsShuffleHeartbeatHandler {
  /** Called when a new peer is seen via heartbeats */
  def addPeer(peer: BlockManagerId): Unit
}

class RapidsShuffleHeartbeatEndpoint(pluginContext: PluginContext, conf: RapidsConf)
  extends Logging with AutoCloseable {
  // Number of milliseconds between heartbeats to driver
  private[this] val heartbeatIntervalMillis =
    conf.shuffleTransportEarlyStartHeartbeatInterval

  private[this] val executorService: ScheduledExecutorService =
    Executors.newSingleThreadScheduledExecutor(new ThreadFactoryBuilder()
      .setNameFormat("rapids-shuffle-hb")
      .setDaemon(true)
      .build())

  private class InitializeShuffleManager(ctx: PluginContext,
      shuffleManager: RapidsShuffleInternalManagerBase) extends Runnable {
    override def run(): Unit = {
      try {
        val serverId = shuffleManager.getServerId
        logInfo(s"Registering executor $serverId with driver")
        ctx.ask(RapidsExecutorStartupMsg(shuffleManager.getServerId)) match {
          case RapidsExecutorUpdateMsg(peers) => updatePeers(shuffleManager, peers)
        }
        val heartbeat = new Runnable {
          override def run(): Unit = {
            try {
              logTrace("Performing executor heartbeat to driver")
              ctx.ask(RapidsExecutorHeartbeatMsg(shuffleManager.getServerId)) match {
                case RapidsExecutorUpdateMsg(peers) => updatePeers(shuffleManager, peers)
              }
            } catch {
              case t: Throwable => logError("Error during heartbeat", t)
            }
          }
        }
        executorService.scheduleWithFixedDelay(
          heartbeat,
          0,
          heartbeatIntervalMillis,
          TimeUnit.MILLISECONDS)
      } catch {
        case t: Throwable => logError("Error initializing shuffle", t)
      }
    }
  }

  def updatePeers(shuffleManager: RapidsShuffleHeartbeatHandler,
      peers: Seq[BlockManagerId]): Unit = {
    peers.foreach { peer =>
      logInfo(s"Updating shuffle manager for new executor $peer")
      shuffleManager.addPeer(peer)
    }
  }

  GpuShuffleEnv.mgr.foreach { mgr =>
    executorService.submit(new InitializeShuffleManager(pluginContext, mgr))
  }

  override def close(): Unit = {
    executorService.shutdown()
  }
}
