"""
Validation-driven RL weight controller for REINVENT4.

This module keeps REINVENT4 core files untouched. It attaches by monkeypatching
two public runtime methods:

1. Scorer.compute_results
   Applies the controller's current weights before REINVENT4 aggregates training
   rewards.

2. Learning.score
   Samples a separate validation batch from the current agent policy and sends
   those SMILES to an independent validation signal provider. Those validation
   scores, not the training reward batch, update the controller policy.

The important separation is:

- Training reward: REINVENT4 scoring components in the TOML config.
- Controller feedback: independent validation scores returned by
  ValidationSignalProvider on a separate validation-only sample.

For quick environment smoke tests, RDKitPropertyValidationSignal can be used.
For the actual DRD2/GSK3B/JNK3 experiments, use an independent QSAR ensemble or
docking wrapper through ExternalCommandValidationSignal.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle
import random
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol


ValidationFn = Callable[[list[str]], dict[str, float]]


def _tensor_bytes(value) -> bytes:
    return value.detach().cpu().contiguous().numpy().tobytes()


def capture_rng_state() -> dict:
    """Capture every process-global RNG stream used by REINVENT sampling."""

    snapshot = {"python": random.getstate(), "numpy": None, "torch_cpu": None, "torch_cuda": None}
    try:
        import numpy as np

        snapshot["numpy"] = np.random.get_state()
    except ImportError:
        pass

    try:
        import torch

        snapshot["torch_cpu"] = torch.get_rng_state().clone()
        if torch.cuda.is_initialized():
            snapshot["torch_cuda"] = [state.clone() for state in torch.cuda.get_rng_state_all()]
    except ImportError:
        pass
    return snapshot


def rng_state_hashes(snapshot: dict) -> dict[str, str | None]:
    """Return stable hashes suitable for controller-history audit records."""

    def digest(value: bytes | None) -> str | None:
        return hashlib.sha256(value).hexdigest() if value is not None else None

    cuda = snapshot.get("torch_cuda")
    return {
        "python": digest(pickle.dumps(snapshot["python"], protocol=4)),
        "numpy": (
            digest(pickle.dumps(snapshot["numpy"], protocol=4))
            if snapshot.get("numpy") is not None
            else None
        ),
        "torch_cpu": (
            digest(_tensor_bytes(snapshot["torch_cpu"]))
            if snapshot.get("torch_cpu") is not None
            else None
        ),
        "torch_cuda": (
            digest(b"".join(_tensor_bytes(state) for state in cuda))
            if cuda is not None
            else None
        ),
    }


def restore_rng_state(snapshot: dict) -> None:
    random.setstate(snapshot["python"])
    if snapshot.get("numpy") is not None:
        import numpy as np

        np.random.set_state(snapshot["numpy"])
    if snapshot.get("torch_cpu") is not None:
        import torch

        torch.set_rng_state(snapshot["torch_cpu"])
        if snapshot.get("torch_cuda") is not None:
            torch.cuda.set_rng_state_all(snapshot["torch_cuda"])


def stable_state_hash(value) -> str:
    """Hash nested model state without relying on checkpoint file serialization."""

    digest = hashlib.sha256()

    def update(item) -> None:
        try:
            import numpy as np
            import torch
        except ImportError:  # pragma: no cover - REINVENT requires these packages
            np = None
            torch = None

        if torch is not None and isinstance(item, torch.Tensor):
            digest.update(b"tensor")
            digest.update(str(item.dtype).encode())
            digest.update(repr(tuple(item.shape)).encode())
            digest.update(_tensor_bytes(item))
        elif np is not None and isinstance(item, np.ndarray):
            digest.update(b"ndarray")
            digest.update(str(item.dtype).encode())
            digest.update(repr(item.shape).encode())
            digest.update(item.tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=lambda value: repr(value)):
                update(key)
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence")
            for child in item:
                update(child)
        elif item is None:
            digest.update(b"none")
        else:
            digest.update(type(item).__qualname__.encode())
            digest.update(repr(item).encode())

    update(value)
    return digest.hexdigest()


def configure_strict_determinism() -> None:
    """Enable deterministic CUDA kernels when a reproducibility test requests it."""

    if os.environ.get("REINVENT_STRICT_DETERMINISM", "0") not in {"1", "true", "TRUE"}:
        return
    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _mean(values: Iterable[float]) -> float:
    vals = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(vals) / len(vals) if vals else 0.0


class ValidationSignalProvider(Protocol):
    """Scores validation-only SMILES with signals independent of training reward."""

    def evaluate(self, smiles: list[str]) -> dict[str, float]:
        """Return one validation score per controller component name."""


@dataclass
class CallableValidationSignal:
    """Adapter for the explicit validation_fn hole used by the controller."""

    validation_fn: ValidationFn

    def evaluate(self, smiles: list[str]) -> dict[str, float]:
        return self.validation_fn(smiles)


@dataclass
class RDKitPropertyValidationSignal:
    """
    Lightweight smoke-test validator.

    This is useful for proving that the controller wiring works. It is not a
    scientifically independent validator if the TOML reward also optimizes the
    same property. For real experiments, replace this with an independent QSAR
    ensemble or docking command provider.
    """

    component_names: list[str]
    mw_low: float = 200.0
    mw_high: float = 500.0

    def evaluate(self, smiles: list[str]) -> dict[str, float]:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, QED

        values: dict[str, list[float]] = {name: [] for name in self.component_names}

        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue

            qed = float(QED.qed(mol))
            mw = float(Descriptors.MolWt(mol))
            mw_score = 1.0 if self.mw_low <= mw <= self.mw_high else 0.0

            for name in self.component_names:
                key = name.lower()
                if key == "qed":
                    values[name].append(qed)
                elif key in {"molecular weight", "mw", "mol_weight"}:
                    values[name].append(mw_score)

        return {name: _mean(values.get(name, [])) for name in self.component_names}


@dataclass
class ExternalCommandValidationSignal:
    """
    Validation provider for independent QSAR/docking jobs.

    The command receives a temporary input CSV with a `smiles` column and must
    write an output CSV with columns matching `component_names`.

    The command template may contain:
      {input}  path to input CSV
      {output} path to output CSV

    Example:
      python validate_qsar.py --input {input} --output {output}
    """

    component_names: list[str]
    command_template: str
    timeout_seconds: int = 3600
    keep_files: bool = False
    workdir: Optional[str] = None

    def evaluate(self, smiles: list[str]) -> dict[str, float]:
        if not smiles:
            return {name: 0.0 for name in self.component_names}

        with tempfile.TemporaryDirectory(prefix="reinvent_validation_") as tmpdir:
            tmp = Path(tmpdir)
            input_csv = tmp / "validation_input.csv"
            output_csv = tmp / "validation_output.csv"

            with input_csv.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["smiles"])
                writer.writeheader()
                for smi in smiles:
                    writer.writerow({"smiles": smi})

            command = self.command_template.format(input=input_csv, output=output_csv)
            subprocess.run(
                command,
                shell=True,
                check=True,
                cwd=self.workdir,
                timeout=self.timeout_seconds,
            )

            scores: dict[str, list[float]] = {name: [] for name in self.component_names}
            with output_csv.open(newline="") as f:
                for row in csv.DictReader(f):
                    for name in self.component_names:
                        if name in row and row[name] not in ("", None):
                            scores[name].append(float(row[name]))

            result = {name: _mean(scores[name]) for name in self.component_names}

            if self.keep_files:
                artifact_dir = Path("validation_artifacts")
                artifact_dir.mkdir(exist_ok=True)
                input_csv.replace(artifact_dir / input_csv.name)
                output_csv.replace(artifact_dir / output_csv.name)

            return result


def run_vina_docking(
    smiles: list[str],
    target: str,
    command_template: Optional[str] = None,
    timeout_seconds: int = 7200,
) -> list[float]:
    """
    Run an external Vina docking pipeline for one target.

    The command receives an input CSV with `smiles` and `target` columns and must
    write an output CSV. Accepted score columns are `vina_score`,
    `docking_score`, `score`, or the target name. Vina affinities are usually
    negative kcal/mol values; the returned list is `-affinity`, so larger is
    better for the controller.

    Template variables:
      {input}  input CSV path
      {output} output CSV path
      {target} target id, e.g. DRD2
    """

    command_template = command_template or os.environ.get("REINVENT_VINA_COMMAND")
    if not command_template:
        raise RuntimeError(
            "No Vina command configured. Set REINVENT_VINA_COMMAND to a command "
            "template such as: python dock_with_vina.py --target {target} "
            "--input {input} --output {output}"
        )

    if not smiles:
        return []

    with tempfile.TemporaryDirectory(prefix=f"vina_{target.lower()}_") as tmpdir:
        tmp = Path(tmpdir)
        input_csv = tmp / "vina_input.csv"
        output_csv = tmp / "vina_output.csv"

        with input_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["smiles", "target"])
            writer.writeheader()
            for smi in smiles:
                writer.writerow({"smiles": smi, "target": target})

        command = command_template.format(input=input_csv, output=output_csv, target=target)
        subprocess.run(command, shell=True, check=True, timeout=timeout_seconds)

        values: list[float] = []
        with output_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                raw = None
                for column in ("vina_score", "docking_score", "score", target):
                    if column in row and row[column] not in ("", None):
                        raw = float(row[column])
                        break
                if raw is not None:
                    values.append(-raw)

        return values


@dataclass
class VinaDockingValidationSignal:
    """
    Independent validation signal for DRD2/GSK3B/JNK3 docking.

    This is the preferred paper-story validator when training reward uses a cheap
    ExCAPE-DB QSAR proxy. The docking batch is sampled only for validation and
    never participates in REINVENT's training reward.
    """

    component_names: list[str]
    command_template: Optional[str] = None
    timeout_seconds: int = 7200
    target_by_component: dict[str, str] = field(default_factory=dict)

    def _target_for_component(self, component_name: str) -> str:
        if component_name in self.target_by_component:
            return self.target_by_component[component_name]
        return component_name.split("_")[0].replace("GSK3β", "GSK3B").upper()

    def evaluate(self, smiles: list[str]) -> dict[str, float]:
        result = {}
        for name in self.component_names:
            target = self._target_for_component(name)
            docking_scores = run_vina_docking(
                smiles,
                target=target,
                command_template=self.command_template,
                timeout_seconds=self.timeout_seconds,
            )
            result[name] = _mean(docking_scores)
        return result


def independent_validator(smilies: list[str]) -> dict[str, float]:
    """
    Example validation_fn for DRD2/GSK3B/JNK3.

    Use this when training reward is a cheap ExCAPE-DB QSAR proxy and controller
    feedback should come from a separate, more faithful oracle such as Vina. Set:

      REINVENT_VALIDATION_TARGETS=DRD2,GSK3B,JNK3
      REINVENT_VINA_COMMAND="python dock_with_vina.py --target {target} --input {input} --output {output}"

    It returns columns named `<TARGET>_activity`, matching common benchmark
    component names.
    """

    targets = [
        target.strip().upper()
        for target in os.environ.get("REINVENT_VALIDATION_TARGETS", "DRD2").split(",")
        if target.strip()
    ]
    scores = {}
    for target in targets:
        docking_scores = run_vina_docking(smilies, target=target)
        scores[f"{target}_activity"] = _mean(docking_scores)
    return scores


@dataclass
class DynamicWeightController:
    """
    Policy-gradient controller for adaptive objective weights.

    At each validation point the controller:

    1. Receives independent validation scores.
    2. Uses the improvement over the previous validation point as reward for
       the previously sampled weight action.
    3. Performs a REINFORCE update with a moving-average baseline.
    4. Samples the next per-objective action and applies it to the weights used
       by future REINVENT4 training steps.
    """

    component_names: list[str]
    validation_fn: Optional[ValidationFn] = None
    validation_provider: Optional[ValidationSignalProvider] = None
    init_weights: dict[str, float] = field(default_factory=dict)
    validate_every: int = 10
    validation_batch_size: int = 128
    validation_max_attempts: int = 4
    action_step: float = 0.15
    min_weight: float = 0.05
    max_weight: float = 5.0
    normalize_weight_sum: bool = True
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    entropy_coef: float = 1e-3
    reward_baseline_beta: float = 0.9
    standardize_advantage: bool = False
    reward_variance_beta: float = 0.9
    min_advantage_std: float = 1e-3
    reward_deadband: float = 0.0
    validation_floor_penalty: float = 0.25
    device: str = "cpu"
    seed: int = 17
    history_path: str = "controller_weight_history.jsonl"

    def __post_init__(self):
        self._weights = {
            name: float(self.init_weights.get(name, 1.0)) for name in self.component_names
        }
        self._initial_weight_sum = sum(self._weights.values()) or 1.0
        self._step_count = 0
        self._validation_count = 0
        self._last_validation_scores: Optional[dict[str, float]] = None
        self._pending_log_prob = None
        self._pending_entropy = None
        self._reward_baseline = 0.0
        self._reward_variance = 1.0
        self._history: list[dict] = []
        self._torch = None
        self._policy = None
        self._optimizer = None
        self._action_generator = None

        if self.validation_fn is None:
            if self.validation_provider is None:
                raise ValueError("Either validation_fn or validation_provider must be provided")
            self.validation_fn = self.validation_provider.evaluate

        self._ensure_policy()

    def _ensure_policy(self):
        if self._policy is not None:
            return

        import torch
        import torch.nn as nn

        self._torch = torch

        num_components = len(self.component_names)
        state_dim = num_components * 3
        action_dim = num_components * 3

        class Policy(nn.Module):
            def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.Tanh(),
                    nn.Linear(hidden_dim, out_dim),
                )

            def forward(self, x):
                return self.net(x)

        cuda_devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.seed)
            self._policy = Policy(state_dim, self.hidden_dim, action_dim).to(self.device)
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=self.learning_rate)
        self._action_generator = torch.Generator(device=self.device)
        self._action_generator.manual_seed(self.seed)

    def get_weights(self) -> dict[str, float]:
        return dict(self._weights)

    def begin_training_step(self) -> bool:
        self._step_count += 1
        return self._step_count % self.validate_every == 0

    def validation_batch_from_learning(self, learning) -> list[str]:
        """Sample validation-only SMILES without changing the training batch."""

        smiles: list[str] = []
        seen: set[str] = set()
        attempts = 0

        while len(smiles) < self.validation_batch_size and attempts < self.validation_max_attempts:
            attempts += 1
            sampled = learning.sampling_model.sample(learning.input_smilies)
            for smi, state in zip(sampled.smilies, sampled.states):
                if str(state).upper().endswith("INVALID"):
                    continue
                if smi in seen:
                    continue
                seen.add(smi)
                smiles.append(smi)
                if len(smiles) >= self.validation_batch_size:
                    break

        return smiles

    def observe_validation_smiles(self, smiles: list[str]) -> dict[str, float]:
        if self.validation_fn is None:
            raise RuntimeError("validation_fn was not configured")
        return self.validation_fn(smiles)

    def sample_and_observe_validation(self, learning) -> tuple[list[str], dict[str, float], dict]:
        """Run the full validation path without advancing training RNG streams."""

        before_state = capture_rng_state()
        before_hashes = rng_state_hashes(before_state)
        try:
            validation_smiles = self.validation_batch_from_learning(learning)
            validation_scores = self.observe_validation_smiles(validation_smiles)
        finally:
            restore_rng_state(before_state)
        after_hashes = rng_state_hashes(capture_rng_state())
        unchanged = before_hashes == after_hashes
        if not unchanged:
            raise RuntimeError("Validation sampling changed a training RNG stream after restoration")
        return validation_smiles, validation_scores, {
            "validation_rng_isolated": True,
            "rng_state_hash_before_validation": before_hashes,
            "rng_state_hash_after_validation": after_hashes,
            "rng_state_unchanged": unchanged,
        }

    def _state_tensor(self, validation_scores: dict[str, float]):
        torch = self._torch
        weight_sum = sum(self._weights.values()) or 1.0

        state_values: list[float] = []
        for name in self.component_names:
            curr = float(validation_scores.get(name, 0.0))
            prev = 0.0
            if self._last_validation_scores is not None:
                prev = float(self._last_validation_scores.get(name, curr))
            state_values.extend(
                [
                    self._weights[name] / weight_sum,
                    curr,
                    curr - prev,
                ]
            )

        return torch.tensor(state_values, dtype=torch.float32, device=self.device)

    def _scalar_controller_reward(self, validation_scores: dict[str, float]) -> float:
        if self._last_validation_scores is None:
            return 0.0

        deltas = []
        floor_penalties = []
        for name in self.component_names:
            curr = float(validation_scores.get(name, 0.0))
            prev = float(self._last_validation_scores.get(name, curr))
            deltas.append(curr - prev)
            floor_penalties.append(max(0.0, prev - curr))

        reward = sum(deltas) / max(1, len(deltas))
        reward -= self.validation_floor_penalty * max(floor_penalties, default=0.0)
        return float(reward)

    def _update_policy(self, reward: float):
        if self._pending_log_prob is None:
            return

        torch = self._torch
        advantage = reward - self._reward_baseline
        self._reward_baseline = (
            self.reward_baseline_beta * self._reward_baseline
            + (1.0 - self.reward_baseline_beta) * reward
        )
        self._reward_variance = (
            self.reward_variance_beta * self._reward_variance
            + (1.0 - self.reward_variance_beta) * (advantage**2)
        )
        if self.standardize_advantage:
            advantage = advantage / math.sqrt(self._reward_variance + self.min_advantage_std**2)
        advantage_t = torch.tensor(advantage, dtype=torch.float32, device=self.device)

        loss = -self._pending_log_prob * advantage_t
        if self._pending_entropy is not None:
            loss = loss - self.entropy_coef * self._pending_entropy

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def _sample_action(self, state):
        torch = self._torch
        logits = self._policy(state).view(len(self.component_names), 3)
        dist = torch.distributions.Categorical(logits=logits)
        action_indices = torch.multinomial(
            dist.probs,
            num_samples=1,
            replacement=True,
            generator=self._action_generator,
        ).squeeze(-1)
        log_prob = dist.log_prob(action_indices).sum()
        entropy = dist.entropy().sum()
        actions = [int(idx.item()) - 1 for idx in action_indices]
        return actions, log_prob, entropy

    def _apply_action(self, actions: list[int]):
        for name, action in zip(self.component_names, actions):
            multiplier = math.exp(self.action_step * float(action))
            self._weights[name] = min(
                self.max_weight,
                max(self.min_weight, self._weights[name] * multiplier),
            )

        if self.normalize_weight_sum:
            current_sum = sum(self._weights.values()) or 1.0
            scale = self._initial_weight_sum / current_sum
            for name in self.component_names:
                self._weights[name] = min(
                    self.max_weight,
                    max(self.min_weight, self._weights[name] * scale),
                )

    def _should_apply_new_action(self, reward: float) -> bool:
        if self._last_validation_scores is None:
            return False
        return abs(float(reward)) > self.reward_deadband

    def step(
        self,
        training_component_scores: Optional[dict[str, float]],
        validation_scores: dict[str, float],
        num_validation_smiles: int = 0,
        rng_audit: Optional[dict] = None,
    ):
        """
        Update the controller from two explicitly separated signal streams.

        training_component_scores comes from REINVENT's normal training reward
        path and is logged for diagnostics only. validation_scores comes from the
        independent validation_fn/oracle and is the only signal used for policy
        reward and weight updates.
        """

        self._validation_count += 1

        reward = self._scalar_controller_reward(validation_scores)
        reward_has_signal = self._should_apply_new_action(reward)
        if reward_has_signal:
            self._update_policy(reward)

        old_weights = dict(self._weights)
        action_applied = reward_has_signal

        if action_applied:
            state = self._state_tensor(validation_scores)
            actions, log_prob, entropy = self._sample_action(state)
            self._apply_action(actions)
        else:
            actions = [0 for _ in self.component_names]
            log_prob = None
            entropy = None

        record = {
            "training_step": self._step_count,
            "validation_index": self._validation_count,
            "num_validation_smiles": num_validation_smiles,
            "training_component_scores": dict(training_component_scores or {}),
            "validation_scores": dict(validation_scores),
            "controller_reward": reward,
            "policy_updated": reward_has_signal,
            "action_applied": action_applied,
            "actions": dict(zip(self.component_names, actions)),
            "old_weights": old_weights,
            "new_weights": dict(self._weights),
            "reward_baseline": self._reward_baseline,
            **dict(rng_audit or {}),
        }
        self._history.append(record)
        self._append_history(record)

        self._last_validation_scores = dict(validation_scores)
        self._pending_log_prob = log_prob
        self._pending_entropy = entropy

    def _append_history(self, record: dict):
        if not self.history_path:
            return
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def history(self) -> list[dict]:
        return list(self._history)


@dataclass
class DeterministicValidationController(DynamicWeightController):
    """
    Deterministic sanity baseline for validation-driven weight updates.

    This controller deliberately removes policy-gradient sampling. If a
    component's independent validation score drops by more than
    `reward_deadband` relative to the previous validation window, that
    component's weight is increased by `action_step`; otherwise it is held. This
    tests whether the validation signal is useful for any simple reactive
    weighting rule before investing further in RL credit assignment.
    """

    decrease_on_improvement: bool = False

    def _ensure_policy(self):
        self._torch = None
        self._policy = None
        self._optimizer = None

    def _deterministic_actions(self, validation_scores: dict[str, float]) -> list[int]:
        if self._last_validation_scores is None:
            return [0 for _ in self.component_names]

        actions: list[int] = []
        for name in self.component_names:
            curr = float(validation_scores.get(name, 0.0))
            prev = float(self._last_validation_scores.get(name, curr))
            delta = curr - prev
            if delta < -self.reward_deadband:
                actions.append(1)
            elif self.decrease_on_improvement and delta > self.reward_deadband:
                actions.append(-1)
            else:
                actions.append(0)
        return actions

    def step(
        self,
        training_component_scores: Optional[dict[str, float]],
        validation_scores: dict[str, float],
        num_validation_smiles: int = 0,
        rng_audit: Optional[dict] = None,
    ):
        self._validation_count += 1

        reward = self._scalar_controller_reward(validation_scores)
        old_weights = dict(self._weights)
        actions = self._deterministic_actions(validation_scores)
        action_applied = any(action != 0 for action in actions)
        if action_applied:
            self._apply_action(actions)

        record = {
            "controller_mode": "deterministic",
            "training_step": self._step_count,
            "validation_index": self._validation_count,
            "num_validation_smiles": num_validation_smiles,
            "training_component_scores": dict(training_component_scores or {}),
            "validation_scores": dict(validation_scores),
            "controller_reward": reward,
            "policy_updated": False,
            "action_applied": action_applied,
            "actions": dict(zip(self.component_names, actions)),
            "old_weights": old_weights,
            "new_weights": dict(self._weights),
            "reward_baseline": self._reward_baseline,
            "deterministic_deadband": self.reward_deadband,
            "decrease_on_improvement": self.decrease_on_improvement,
            **dict(rng_audit or {}),
        }
        self._history.append(record)
        self._append_history(record)

        self._last_validation_scores = dict(validation_scores)
        self._pending_log_prob = None
        self._pending_entropy = None


@dataclass
class FrozenWeightValidationController(DynamicWeightController):
    """Validation-compute-matched static control with permanently fixed weights."""

    def _ensure_policy(self):
        self._torch = None
        self._policy = None
        self._optimizer = None
        self._action_generator = None

    def step(
        self,
        training_component_scores: Optional[dict[str, float]],
        validation_scores: dict[str, float],
        num_validation_smiles: int = 0,
        rng_audit: Optional[dict] = None,
    ):
        self._validation_count += 1
        old_weights = dict(self._weights)
        record = {
            "controller_mode": "frozen_weight",
            "frozen_weight_control": True,
            "training_step": self._step_count,
            "validation_index": self._validation_count,
            "num_validation_smiles": num_validation_smiles,
            "training_component_scores": dict(training_component_scores or {}),
            "validation_scores": dict(validation_scores),
            "controller_reward": self._scalar_controller_reward(validation_scores),
            "policy_updated": False,
            "action_applied": False,
            "actions": {name: 0 for name in self.component_names},
            "old_weights": old_weights,
            "new_weights": dict(self._weights),
            "reward_baseline": self._reward_baseline,
            **dict(rng_audit or {}),
        }
        self._history.append(record)
        self._append_history(record)

        self._last_validation_scores = dict(validation_scores)
        self._pending_log_prob = None
        self._pending_entropy = None


def attach_controller(controller: DynamicWeightController):
    """Attach the controller without changing REINVENT4 source files."""

    from reinvent.scoring.scorer import Scorer
    from reinvent.runmodes.RL.learning import Learning

    if getattr(Scorer, "_dynamic_weight_controller_patched", False):
        raise RuntimeError("Scorer is already patched with a dynamic weight controller")

    original_compute_results = Scorer.compute_results
    original_learning_score = Learning.score
    original_learning_report = Learning.report
    training_trace_path = os.environ.get("REINVENT_TRAINING_TRACE_PATH")

    def _extract_component_scores(results) -> dict[str, float]:
        component_scores = {}
        for comp in results.completed_components:
            for name, tscores in zip(comp.component_names, comp.transformed_scores):
                if len(tscores) > 0:
                    component_scores[name] = float(sum(tscores) / len(tscores))
        return component_scores

    def _find_weight_slot(scorer_self, name: str):
        for component in scorer_self.components.scorers:
            names, _scoring_fn, _transforms, weights = component.params
            if name in names:
                idx = names.index(name)
                return weights, idx
        return None, None

    def _apply_weights(scorer_self, weights: dict[str, float]):
        for name, weight in weights.items():
            weight_list, idx = _find_weight_slot(scorer_self, name)
            if weight_list is not None:
                weight_list[idx] = float(weight)

    def _append_training_trace(record: dict):
        if not training_trace_path:
            return
        with open(training_trace_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")

    def _append_training_score_trace(learning, results, training_component_scores: dict[str, float]):
        if not training_trace_path:
            return
        total_scores = getattr(results, "total_scores", [])
        _append_training_trace({
            "trace_kind": "training_score",
            "training_step": controller._step_count,
            "smilies": [str(value) for value in learning.sampled.smilies],
            "smiles_states": [str(value) for value in learning.sampled.states],
            "total_scores": [float(value) for value in total_scores],
            "training_component_scores": dict(training_component_scores),
            "rng_state_hash_after_training_score": rng_state_hashes(capture_rng_state()),
        })

    def patched_learning_report(self, *args, **kwargs):
        result = original_learning_report(self, *args, **kwargs)
        if args:
            step = int(args[0]) + 1
        else:
            step = int(kwargs["step"]) + 1
        if training_trace_path:
            _append_training_trace({
                "trace_kind": "agent_update",
                "training_step": step,
                "post_update_agent_state_hash": stable_state_hash(
                    self._state.agent.get_save_dict()["network"]
                ),
                "rng_state_hash_after_agent_update": rng_state_hashes(capture_rng_state()),
            })
        return result

    def patched_compute_results(
        self,
        smilies,
        invalid_mask,
        duplicate_mask,
        fragments=None,
        connectivity_annotated_smiles=None,
    ):
        _apply_weights(self, controller.get_weights())
        return original_compute_results(
            self,
            smilies,
            invalid_mask,
            duplicate_mask,
            fragments,
            connectivity_annotated_smiles,
        )

    def patched_learning_score(self):
        should_validate = controller.begin_training_step()
        results = original_learning_score(self)
        training_component_scores = _extract_component_scores(results)
        _append_training_score_trace(self, results, training_component_scores)

        if should_validate:
            validation_smiles, validation_scores, rng_audit = (
                controller.sample_and_observe_validation(self)
            )
            controller.step(
                training_component_scores=training_component_scores,
                validation_scores=validation_scores,
                num_validation_smiles=len(validation_smiles),
                rng_audit=rng_audit,
            )

        return results

    Scorer.compute_results = patched_compute_results
    Scorer.__call__ = patched_compute_results
    Scorer._dynamic_weight_controller_patched = True

    Learning.score = patched_learning_score
    Learning._dynamic_weight_controller_patched = True
    Learning.report = patched_learning_report

    return controller


def build_default_controller() -> DynamicWeightController:
    def parse_bool(name: str, default: str = "0") -> bool:
        return os.environ.get(name, default) in {"1", "true", "TRUE", "yes", "YES"}

    def parse_component_names() -> list[str]:
        raw = os.environ.get("REINVENT_CONTROLLER_COMPONENTS", "QED,Molecular weight")
        names = [name.strip() for name in raw.split(",") if name.strip()]
        if not names:
            raise ValueError("REINVENT_CONTROLLER_COMPONENTS did not contain any component names")
        return names

    def parse_init_weights(names: list[str]) -> dict[str, float]:
        raw = os.environ.get("REINVENT_CONTROLLER_INIT_WEIGHTS", "")
        if not raw.strip():
            equal = 1.0 / max(1, len(names))
            return {name: equal for name in names}

        values = [value.strip() for value in raw.split(",") if value.strip()]
        if len(values) != len(names):
            raise ValueError(
                "REINVENT_CONTROLLER_INIT_WEIGHTS must have one comma-separated "
                "value per REINVENT_CONTROLLER_COMPONENTS entry"
            )
        return {name: float(value) for name, value in zip(names, values)}

    component_names = parse_component_names()
    validation_command = os.environ.get("REINVENT_VALIDATION_COMMAND")
    init_weights = parse_init_weights(component_names)
    controller_mode = os.environ.get("REINVENT_CONTROLLER_MODE", "rl").strip().lower()

    if validation_command:
        provider: ValidationSignalProvider = ExternalCommandValidationSignal(
            component_names=component_names,
            command_template=validation_command,
        )
    elif os.environ.get("REINVENT_VINA_COMMAND"):
        provider = VinaDockingValidationSignal(
            component_names=component_names,
            command_template=os.environ["REINVENT_VINA_COMMAND"],
        )
    else:
        provider = RDKitPropertyValidationSignal(component_names=component_names)

    controller_kwargs = dict(
        component_names=component_names,
        validation_fn=provider.evaluate,
        init_weights=init_weights,
        validate_every=int(os.environ.get("REINVENT_CONTROLLER_VALIDATE_EVERY", "5")),
        validation_batch_size=int(os.environ.get("REINVENT_CONTROLLER_VALIDATION_BATCH", "64")),
        action_step=float(os.environ.get("REINVENT_CONTROLLER_ACTION_STEP", "0.15")),
        learning_rate=float(os.environ.get("REINVENT_CONTROLLER_LR", "0.001")),
        entropy_coef=float(os.environ.get("REINVENT_CONTROLLER_ENTROPY_COEF", "0.001")),
        reward_baseline_beta=float(os.environ.get("REINVENT_CONTROLLER_REWARD_BASELINE_BETA", "0.9")),
        standardize_advantage=parse_bool("REINVENT_CONTROLLER_STANDARDIZE_ADVANTAGE"),
        reward_variance_beta=float(os.environ.get("REINVENT_CONTROLLER_REWARD_VARIANCE_BETA", "0.9")),
        reward_deadband=float(os.environ.get("REINVENT_CONTROLLER_REWARD_DEADBAND", "0.0")),
        validation_floor_penalty=float(os.environ.get("REINVENT_CONTROLLER_FLOOR_PENALTY", "0.25")),
        device=os.environ.get("REINVENT_CONTROLLER_DEVICE", "cpu"),
        seed=int(os.environ.get("REINVENT_CONTROLLER_SEED", "17")),
        history_path=os.environ.get(
            "REINVENT_CONTROLLER_HISTORY_PATH",
            "controller_weight_history.jsonl",
        ),
    )

    if controller_mode in {"deterministic", "rule", "rule_based"}:
        return DeterministicValidationController(
            **controller_kwargs,
            decrease_on_improvement=parse_bool(
                "REINVENT_CONTROLLER_DECREASE_ON_IMPROVEMENT",
                "0",
            ),
        )
    if controller_mode in {"frozen", "frozen_weight", "fixed_validation"}:
        return FrozenWeightValidationController(**controller_kwargs)
    if controller_mode not in {"rl", "policy", "policy_gradient"}:
        raise ValueError(
            "REINVENT_CONTROLLER_MODE must be one of: rl, deterministic, frozen_weight"
        )

    return DynamicWeightController(**controller_kwargs)


if __name__ == "__main__":
    import sys

    configure_strict_determinism()
    controller = build_default_controller()
    attach_controller(controller)

    from reinvent.Reinvent import main_script

    if len(sys.argv) == 1:
        sys.argv += ["-l", "controlled_run.log", "02_baseline_multiobj.toml"]

    main_script()
