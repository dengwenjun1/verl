# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

from omegaconf import DictConfig

from verl.protocol import DataProto
from verl.utils.skip.base_skip import BaseSkip, SkipAction, register_skip
from verl.utils.skip.config import SkipManagerConfig


def get_training_skip_metrics(config: DictConfig | SkipManagerConfig | None, step: int) -> dict[str, int]:
    """Report training skips for an explicit trainer step, including delayed metric collection.

    Return no metrics when both roles are disabled. Skipping train_compute also bypasses
    its nested updates, regardless of the updates role's own step selection.
    """
    if config is None:
        return {}
    updates = config.get("updates")
    train_compute = config.get("train_compute")
    if not any(role is not None and role.enable for role in (updates, train_compute)):
        return {}

    def selected(role) -> bool:
        return role is not None and role.enable and step >= 1 and (role.steps is None or step in role.steps)

    skip_compute = selected(train_compute)
    return {
        "training/skip_updates": int(selected(updates) or skip_compute),
        "training/skip_train_compute": int(skip_compute),
    }


@register_skip("updates")
@register_skip("train_compute")
class TrainingSkip(BaseSkip):
    """Bypass training stages, preserving the batch and all surrounding lifecycle hooks."""

    support_actions = [SkipAction.EMPTY]

    def meet_precondition(self, step, func, trainer, batch, *args, **kwargs) -> bool:
        """Support V1 queue metadata and V0 DataProto training batches."""
        if step < 1:
            return False
        if isinstance(batch, DataProto):
            return not batch.meta_info.get("validate", False)
        return getattr(batch, "partition_id", None) == "train"

    def warp_function(self, step, func, trainer, batch, *args, **kwargs):
        """Return the existing batch without training, modifying queue data, or disk I/O."""
        return batch

    def prepare_data(self, step, result, *args, **kwargs):
        """Training skip does not persist or replay results."""
