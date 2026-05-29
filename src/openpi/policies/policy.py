from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils
from scipy.spatial.transform import Rotation as R
from progress_tracker.progress_tracker import ProgressTracker
import os

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        save_state_dir: str | None = None,
        guidance_config: dict[str, Any] | None = None,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            save_state_dir: Directory to save the state of the policy.
            guidance_config: Configuration for guidance.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._save_state_dir = save_state_dir
        if self._save_state_dir is not None:
            os.makedirs(self._save_state_dir, exist_ok=True)
        self._guidance_config = guidance_config
        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

        self._model.save_state_dir = self._save_state_dir

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if "observation/cube" in obs:
            inputs["cube"] = obs["observation/cube"]
        if "observation/left_image" in obs:
            inputs["left_image"] = obs["observation/left_image"].astype(np.float32) / 255.0
            inputs["left_intrin"] = obs["observation/left_intrin"]
            inputs["left_extrin"] = obs["observation/left_extrin"]
        if "observation/right_image" in obs:
            inputs["right_image"] = obs["observation/right_image"].astype(np.float32) / 255.0
            inputs["right_intrin"] = obs["observation/right_intrin"]
            inputs["right_extrin"] = obs["observation/right_extrin"]
        if "observation/overhead_image" in obs:
            inputs["overhead_image"] = obs["observation/overhead_image"].astype(np.float32) / 255.0
            inputs["overhead_intrin"] = obs["observation/overhead_intrin"]
            inputs["overhead_extrin"] = obs["observation/overhead_extrin"]
        if "observation/wrist_image" in obs:
            inputs["wrist_image"] = obs["observation/wrist_image"].astype(np.float32) / 255.0
        if "observation/left_depth" in obs:
            inputs["left_depth"] = obs["observation/left_depth"]
            inputs["right_depth"] = obs["observation/right_depth"]
            inputs["overhead_depth"] = obs["observation/overhead_depth"]
            
        inputs["joint_position"] = obs["observation/joint_position"]
        inputs["gripper_position"] = obs["observation/gripper_position"]
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        sample_kwargs["output_transform"] = self._output_transform

        # Preserve the raw prompt text if available
        if "prompt" in obs:
            sample_kwargs["prompt"] = obs["prompt"]

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {"state": inputs["state"], "actions": actions}
        model_time = time.monotonic() - start_time

        # Save observation and actions together with sample data
        if self._save_state_dir is not None and self._is_pytorch_model:
            save_dict = {
                "observation": obs,
                # "actions": actions.cpu().numpy() if isinstance(actions, torch.Tensor) else np.asarray(actions),
                "model_time_ms": model_time * 1000,
            }
            np.save(f"{self._save_state_dir}/observations/infer_{self._model.global_count}.npy", save_dict)
            print(
                f"[SAVE] Saved inference data to {self._save_state_dir}/observations/infer_{self._model.global_count}.npy"
            )
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)  # ['actions'] = (15, 32)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
