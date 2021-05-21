# Copyright (c) 2021, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import time
import threading
from spark_session import with_gpu_session
from marks import allow_non_gpu, ucx
from conftest import _get_jvm

_conf = {
}

def thread_function(spark):
  jvm = _get_jvm(spark)
  UCX = jvm.com.nvidia.spark.rapids.shuffle.ucx.UCX
  while True:
    UCX.doProgress()
    time.sleep(1)
    print("here")

@ucx
def test_ucx_simple(enable_ucx):
    def do_work(spark):
      jsc = spark._jsc.sc()
      print(jsc.getExecutorIds())
      df = spark.read.parquet("file:///home/abellina/foo")
      x = threading.Thread(target=thread_function, args=(spark,))
      x.start()
      print(df.repartition(1000).collect())

    with_gpu_session(
      lambda spark: do_work(spark),
      conf = _conf)
