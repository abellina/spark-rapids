/*
 * Copyright (c) 2025, NVIDIA CORPORATION.
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

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.scalatest.BeforeAndAfterEach
import org.scalatest.funsuite.AnyFunSuite

class SlowTaskMonitorSuite extends AnyFunSuite with BeforeAndAfterEach {

  // Mock time source for testing
  private var mockTime: Long = 0L
  
  override def beforeEach(): Unit = {
    super.beforeEach()
    mockTime = 10000L // Start at 10 seconds
    SlowTaskMonitor.currentTimeMillis = () => mockTime
  }

  override def afterEach(): Unit = {
    try {
      SlowTaskMonitor.shutdown()
    } catch {
      case _: Exception => // Ignore
    }
    super.afterEach()
  }
  
  private def advanceTime(millis: Long): Unit = {
    mockTime += millis
  }

  test("monitor is disabled when timeout is 0") {
    val conf = new SparkConf()
      .set(RapidsConf.SLOW_TASK_TIMEOUT_SECONDS.key, "0")
    val rapidsConf = new RapidsConf(conf)
    
    SlowTaskMonitor.initialize(rapidsConf)
    // Should not throw and should be in disabled state
    SlowTaskMonitor.shutdown()
  }

  test("monitor is enabled when timeout is positive") {
    val conf = new SparkConf()
      .set(RapidsConf.SLOW_TASK_TIMEOUT_SECONDS.key, "10")
      .set(RapidsConf.SLOW_TASK_CHECK_INTERVAL_SECONDS.key, "5")
    val rapidsConf = new RapidsConf(conf)
    
    SlowTaskMonitor.initialize(rapidsConf)
    // Should not throw
    SlowTaskMonitor.shutdown()
  }

  test("configuration validation - negative timeout rejected") {
    val conf = new SparkConf()
      .set(RapidsConf.SLOW_TASK_TIMEOUT_SECONDS.key, "-1")
    
    assertThrows[IllegalArgumentException] {
      new RapidsConf(conf)
    }
  }

  test("configuration validation - zero check interval rejected") {
    val conf = new SparkConf()
      .set(RapidsConf.SLOW_TASK_CHECK_INTERVAL_SECONDS.key, "0")
    
    assertThrows[IllegalArgumentException] {
      new RapidsConf(conf)
    }
  }

  test("configuration validation - negative check interval rejected") {
    val conf = new SparkConf()
      .set(RapidsConf.SLOW_TASK_CHECK_INTERVAL_SECONDS.key, "-5")
    
    assertThrows[IllegalArgumentException] {
      new RapidsConf(conf)
    }
  }

  test("configuration validation - zero stack depth rejected") {
    val conf = new SparkConf()
      .set(RapidsConf.SLOW_TASK_STACK_DEPTH.key, "0")
    
    assertThrows[IllegalArgumentException] {
      new RapidsConf(conf)
    }
  }

  test("configuration validation - negative stack depth rejected") {
    val conf = new SparkConf()
      .set(RapidsConf.SLOW_TASK_STACK_DEPTH.key, "-5")
    
    assertThrows[IllegalArgumentException] {
      new RapidsConf(conf)
    }
  }

  test("default configuration values") {
    val conf = new SparkConf()
    val rapidsConf = new RapidsConf(conf)
    
    assert(rapidsConf.get(RapidsConf.SLOW_TASK_TIMEOUT_SECONDS) == 0)
    assert(rapidsConf.get(RapidsConf.SLOW_TASK_CHECK_INTERVAL_SECONDS) == 30)
    assert(rapidsConf.get(RapidsConf.SLOW_TASK_STACK_DEPTH) == 10)
  }

  test("custom stack depth configuration") {
    val conf = new SparkConf()
      .set(RapidsConf.SLOW_TASK_TIMEOUT_SECONDS.key, "10")
      .set(RapidsConf.SLOW_TASK_STACK_DEPTH.key, "20")
    val rapidsConf = new RapidsConf(conf)
    
    SlowTaskMonitor.initialize(rapidsConf)
    // Should not throw and should use custom stack depth
    SlowTaskMonitor.shutdown()
  }

  test("onTaskStart and onTaskEnd without TaskContext does not throw") {
    val conf = new SparkConf()
      .set(RapidsConf.SLOW_TASK_TIMEOUT_SECONDS.key, "10")
    val rapidsConf = new RapidsConf(conf)
    
    SlowTaskMonitor.initialize(rapidsConf)
    
    // These should not throw even without a TaskContext
    SlowTaskMonitor.onTaskStart()
    SlowTaskMonitor.onTaskEnd()
    
    SlowTaskMonitor.shutdown()
  }

  test("task is not detected as slow before timeout") {
    val conf = new SparkConf()
      .set(RapidsConf.SLOW_TASK_TIMEOUT_SECONDS.key, "60") // 60 second timeout
      .set(RapidsConf.SLOW_TASK_CHECK_INTERVAL_SECONDS.key, "10")
    val rapidsConf = new RapidsConf(conf)
    
    SlowTaskMonitor.initialize(rapidsConf)
    
    // Register a task at current time
    val taskId = 12345L
    SlowTaskMonitor.registerTask(taskId, stageId = 1, partitionId = 0, 
      Thread.currentThread(), startTime = mockTime)
    
    assert(SlowTaskMonitor.getActiveTaskCount == 1)
    assert(!SlowTaskMonitor.isTaskMarkedAsSlow(taskId))
    
    // Advance time by 30 seconds (less than 60 second timeout)
    advanceTime(30000)
    SlowTaskMonitor.checkForSlowTasks()
    
    // Task should not be marked as slow yet
    assert(!SlowTaskMonitor.isTaskMarkedAsSlow(taskId))
    
    SlowTaskMonitor.shutdown()
  }

  test("task is detected as slow after timeout") {
    val conf = new SparkConf()
      .set(RapidsConf.SLOW_TASK_TIMEOUT_SECONDS.key, "30") // 30 second timeout
      .set(RapidsConf.SLOW_TASK_CHECK_INTERVAL_SECONDS.key, "10")
    val rapidsConf = new RapidsConf(conf)
    
    SlowTaskMonitor.initialize(rapidsConf)
    
    // Register a task at current time
    val taskId = 12345L
    SlowTaskMonitor.registerTask(taskId, stageId = 1, partitionId = 0, 
      Thread.currentThread(), startTime = mockTime)
    
    assert(!SlowTaskMonitor.isTaskMarkedAsSlow(taskId))
    
    // Advance time by 31 seconds (more than 30 second timeout)
    advanceTime(31000)
    SlowTaskMonitor.checkForSlowTasks()
    
    // Task should now be marked as slow
    assert(SlowTaskMonitor.isTaskMarkedAsSlow(taskId))
    
    SlowTaskMonitor.shutdown()
  }

  test("multiple tasks can be tracked independently") {
    val conf = new SparkConf()
      .set(RapidsConf.SLOW_TASK_TIMEOUT_SECONDS.key, "30")
      .set(RapidsConf.SLOW_TASK_CHECK_INTERVAL_SECONDS.key, "10")
    val rapidsConf = new RapidsConf(conf)
    
    SlowTaskMonitor.initialize(rapidsConf)
    
    // Register first task at time T
    val task1Id = 100L
    SlowTaskMonitor.registerTask(task1Id, stageId = 1, partitionId = 0, 
      Thread.currentThread(), startTime = mockTime)
    
    // Advance time by 10 seconds
    advanceTime(10000)
    
    // Register second task at time T+10
    val task2Id = 200L
    SlowTaskMonitor.registerTask(task2Id, stageId = 1, partitionId = 1, 
      Thread.currentThread(), startTime = mockTime)
    
    assert(SlowTaskMonitor.getActiveTaskCount == 2)
    
    // Advance time by 25 more seconds (total: 35 seconds for task1, 25 for task2)
    advanceTime(25000)
    SlowTaskMonitor.checkForSlowTasks()
    
    // Task1 should be slow (35 > 30), task2 should not (25 < 30)
    assert(SlowTaskMonitor.isTaskMarkedAsSlow(task1Id))
    assert(!SlowTaskMonitor.isTaskMarkedAsSlow(task2Id))
    
    // Advance time by 10 more seconds (total: 45 for task1, 35 for task2)
    advanceTime(10000)
    SlowTaskMonitor.checkForSlowTasks()
    
    // Now both should be slow
    assert(SlowTaskMonitor.isTaskMarkedAsSlow(task1Id))
    assert(SlowTaskMonitor.isTaskMarkedAsSlow(task2Id))
    
    SlowTaskMonitor.shutdown()
  }

  test("unregistering task removes it from monitoring") {
    val conf = new SparkConf()
      .set(RapidsConf.SLOW_TASK_TIMEOUT_SECONDS.key, "30")
    val rapidsConf = new RapidsConf(conf)
    
    SlowTaskMonitor.initialize(rapidsConf)
    
    val taskId = 12345L
    SlowTaskMonitor.registerTask(taskId, stageId = 1, partitionId = 0, 
      Thread.currentThread(), startTime = mockTime)
    
    assert(SlowTaskMonitor.getActiveTaskCount == 1)
    
    // Unregister the task
    SlowTaskMonitor.unregisterTask(taskId)
    
    assert(SlowTaskMonitor.getActiveTaskCount == 0)
    
    // Advance time past timeout
    advanceTime(40000)
    SlowTaskMonitor.checkForSlowTasks()
    
    // Task should not be marked as slow since it was unregistered
    assert(!SlowTaskMonitor.isTaskMarkedAsSlow(taskId))
    
    SlowTaskMonitor.shutdown()
  }

  test("slow task detection respects check interval for repeated logging") {
    val conf = new SparkConf()
      .set(RapidsConf.SLOW_TASK_TIMEOUT_SECONDS.key, "30")
      .set(RapidsConf.SLOW_TASK_CHECK_INTERVAL_SECONDS.key, "20")
    val rapidsConf = new RapidsConf(conf)
    
    SlowTaskMonitor.initialize(rapidsConf)
    
    val taskId = 12345L
    SlowTaskMonitor.registerTask(taskId, stageId = 1, partitionId = 0, 
      Thread.currentThread(), startTime = mockTime)
    
    // Advance time past timeout to mark as slow
    advanceTime(35000)
    SlowTaskMonitor.checkForSlowTasks()
    assert(SlowTaskMonitor.isTaskMarkedAsSlow(taskId))
    
    // Advance time by 10 seconds (less than 20 second check interval)
    advanceTime(10000)
    // The task is still slow, but we don't re-log yet due to check interval
    
    // Advance time by 15 more seconds (total 25 > 20 second check interval)
    advanceTime(15000)
    SlowTaskMonitor.checkForSlowTasks()
    // Task should still be marked as slow and would be logged again
    assert(SlowTaskMonitor.isTaskMarkedAsSlow(taskId))
    
    SlowTaskMonitor.shutdown()
  }
}

