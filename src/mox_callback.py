# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""callback function"""
from mindspore.train.callback import Callback



class MoxingCallBack(Callback):
    """EvaluateCallBack"""

    def __init__(self, src_url, train_url, total_epochs, save_freq=2):
        super(MoxingCallBack, self).__init__()

        self.src_url = src_url
        self.train_url = train_url
        self.total_epochs = total_epochs
        self.save_freq = save_freq

    def epoch_end(self, run_context):
        """
            Test when epoch end, save best model with best.ckpt.
        """
        cb_params = run_context.original_args()
        # if cb_params.cur_epoch_num > self.total_epochs * 0.9 or int(
        #         cb_params.cur_epoch_num - 1) % 10 == 0 or cb_params.cur_epoch_num < 10:
        cur_epoch_num = cb_params.cur_epoch_num
        import moxing as mox
        if cur_epoch_num % self.save_freq == 0:
            mox.file.copy_parallel(src_url=self.src_url, dst_url=self.train_url)
