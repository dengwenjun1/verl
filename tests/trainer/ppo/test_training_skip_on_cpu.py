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

"""Exercise training skip decorators at the real V0 and V1 trainer boundaries."""

from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf

from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.v1.trainer_base import PPOTrainer
from verl.utils.skip import SkipManager
from verl.utils.skip.config import TrainingSkipConfig
from verl.utils.skip.training_skip import TrainingSkip


@pytest.fixture(autouse=True)
def reset_skip_state(monkeypatch):
    monkeypatch.setattr(SkipManager, "step", 1)
    monkeypatch.setattr(SkipManager, "skip_instances", {})


@pytest.fixture(params=[RayPPOTrainer, PPOTrainer], ids=["v0", "v1"])
def trainer_case(request):
    trainer_class = request.param
    batch = (
        DataProto.from_dict(tensors={"x": torch.ones(1)})
        if trainer_class is RayPPOTrainer
        else SimpleNamespace(partition_id="train", keys=["sample"], tags=[])
    )
    output = DataProto(meta_info={"metrics": {}}) if trainer_class is RayPPOTrainer else batch
    trainer = SimpleNamespace(
        use_critic=True,
        global_steps=1,
        config=OmegaConf.create({"trainer": {"critic_warmup": 0}}),
        _update_actor=Mock(return_value=output),
        _update_critic=Mock(return_value=output),
    )
    trainer._update_train_batch = MethodType(trainer_class._update_train_batch, trainer)
    return trainer_class, trainer, batch


@pytest.mark.parametrize("steps, expected_calls", [([1], 0), ([2], 1)], ids=["selected", "unselected"])
def test_updates_role_intercepts_real_update_boundary(trainer_case, steps, expected_calls):
    _, trainer, batch = trainer_case
    SkipManager.skip_instances["updates"] = TrainingSkip(TrainingSkipConfig(enable=True, steps=steps), None)
    metrics = {}

    result = trainer._update_train_batch(batch, metrics, {})

    assert result is batch
    assert trainer._update_actor.call_count == expected_calls
    assert trainer._update_critic.call_count == expected_calls
    assert metrics == {}


def test_train_compute_alone_skips_computation_and_updates(trainer_case):
    trainer_class, trainer, batch = trainer_case
    SkipManager.skip_instances["train_compute"] = TrainingSkip(TrainingSkipConfig(enable=True), None)
    trainer._update_train_batch = Mock(side_effect=AssertionError("updates must not run without training inputs"))
    metrics = {}
    timing = {}

    result = trainer_class._train_batch(trainer, batch, metrics, timing)

    assert result is batch
    trainer._update_train_batch.assert_not_called()
    trainer._update_actor.assert_not_called()
    trainer._update_critic.assert_not_called()
    assert metrics == {}
    assert timing == {}


def test_unselected_updates_preserve_critic_warmup(trainer_case):
    _, trainer, batch = trainer_case
    trainer.config.trainer.critic_warmup = 2
    SkipManager.skip_instances["updates"] = TrainingSkip(TrainingSkipConfig(enable=True, steps=[2]), None)

    assert trainer._update_train_batch(batch, {}, {}) is batch
    trainer._update_actor.assert_not_called()
    trainer._update_critic.assert_called_once()


def test_v1_updates_skip_keeps_training_input_computation():
    SkipManager.skip_instances["updates"] = TrainingSkip(TrainingSkipConfig(enable=True), None)
    batch = SimpleNamespace(partition_id="train")
    compute_methods = [
        "_balance_batch",
        "_compute_old_log_prob",
        "_compute_ref_log_prob",
        "_compute_values",
        "_compute_advantage",
    ]
    trainer = SimpleNamespace(use_reference_policy=True, use_critic=True)
    for method in compute_methods:
        setattr(trainer, method, Mock(return_value=batch))
    trainer._update_actor = Mock(side_effect=AssertionError("actor update must be skipped"))
    trainer._update_critic = Mock(side_effect=AssertionError("critic update must be skipped"))
    trainer._update_train_batch = MethodType(PPOTrainer._update_train_batch, trainer)

    assert PPOTrainer._train_batch(trainer, batch, {}, {}) is batch
    for method in compute_methods:
        getattr(trainer, method).assert_called_once()
    trainer._update_actor.assert_not_called()
    trainer._update_critic.assert_not_called()
