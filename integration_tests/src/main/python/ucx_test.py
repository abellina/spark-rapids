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
from spark_session import with_gpu_session
from marks import allow_non_gpu, ucx

_conf = {
}

@ucx
def test_ucx_simple(enable_ucx):
    def do_work(spark):
      sc = spark._jsc.sc()
      print(sc.getExecutorIds())

    with_gpu_session(
      lambda spark: do_work(spark),
      conf = _conf)
