import numpy as np
import requests
import torch

from collision.vggt_pcd import preprocess_images


class ProgressTracker:
    def __init__(self, base_url, max_horizon=1, img_keys=["left_image", "wrist_image"],
                 grasping_threshold=0.3, releasing_threshold=0.1):
        self.base_url = base_url
        self.max_horizon = max_horizon
        self.obs_queue = {}
        self.keys = img_keys
        self.grasping_threshold = grasping_threshold
        self.releasing_threshold = releasing_threshold

        self.subtasks = None
        self.current_subtask = 0

    def _extract_gripper_scalar(self, gripper_position) -> float:
        """Normalize gripper position to a scalar float."""
        if torch.is_tensor(gripper_position):
            gp = gripper_position.detach().cpu().numpy().reshape(-1)
        else:
            gp = np.asarray(gripper_position).reshape(-1)
        assert gp.size > 0, "gripper_position is empty"
        return float(gp[0])

    def set_subtasks(self, prompt: str):
        if self.subtasks is not None:
            return
        pay_load = {"task_description": prompt}
        r = requests.post(f"{self.base_url}/decompose", json=pay_load, timeout=30)
        print("decompose result:", r.json())
        self.subtasks = r.json()["result"]["subtasks"]

    def process_obs(self, obs, downsample=True):
        """Process observation and convert to JSON-serializable format.

        Each image is expected to be shape (1, 3, H, W) as torch tensor.
        We convert to (H, W, 3) numpy array in uint8 format.
        """
        processed = {}
        for key in self.keys:
            if hasattr(obs, key):
                img_tensor = getattr(obs, key)
                if img_tensor is not None:
                    if downsample:
                        img_tensor = preprocess_images(img_tensor)
                    img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
                    img_uint8 = (img_np * 255).astype(np.uint8)
                    processed[key] = img_uint8.tolist()
        processed["gripper_position"] = self._extract_gripper_scalar(obs.gripper_position)
        return processed

    def save_obs(self, obs, global_count, downsample=True):
        self.obs_queue[global_count] = self.process_obs(obs, downsample=downsample)
        if len(self.obs_queue) > self.max_horizon:
            self.obs_queue.pop(min(self.obs_queue.keys()))

    def check_completion(self, subtask: str, just_exit_freezing_zone=None):
        """Check if a subtask is completed.

        Args:
            subtask: The subtask description to check
            just_exit_freezing_zone: Optional boolean to control VLM calling
                - None or True: Check completion (call VLM)
                - False: Skip VLM call, return not completed

        Returns:
            dict: Result with 'completed' status
        """
        latest_obs = self.obs_queue[max(self.obs_queue.keys())]
        gripper_position = self._extract_gripper_scalar(latest_obs["gripper_position"])
        payload = {
            "task_description": subtask,
            "gripper_position": gripper_position,
            "trajectory": self.obs_queue,
        }
        # Heuristic pre-filter: skip expensive VLM call when gripper state
        # clearly contradicts the expected completion condition.
        grasping_tasks = ["pick", "grasp"]
        releasing_tasks = ["put", "pull"]
        if any(e in subtask for e in grasping_tasks) and gripper_position < self.grasping_threshold:
            print(f"[PROGRESS TRACKER] Skipping grasping subtask: `{subtask}` because gripper position is less than {self.grasping_threshold}")
            return {"result": {"completed": False}}
        if any(e in subtask for e in releasing_tasks) and gripper_position > self.releasing_threshold:
            print(f"[PROGRESS TRACKER] Skipping releasing subtask: `{subtask}` because gripper position is greater than {self.releasing_threshold}")
            return {"result": {"completed": False}}
        print(f"[PROGRESS TRACKER] Requesting subtask: `{subtask}` | gripper position: {gripper_position}")
        r = requests.post(f"{self.base_url}/predict", json=payload, timeout=120)
        if not r.ok:
            print(f"[PROGRESS TRACKER] /predict failed with status {r.status_code}: {r.text[:300]}")
            return {"result": {"completed": False}, "success": False}
        content_type = r.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            print(
                f"[PROGRESS TRACKER] /predict returned non-JSON response ({content_type}): {r.text[:300]}"
            )
            return {"result": {"completed": False}, "success": False}
        result = r.json()
        print("predict result:", result)
        return result

    def get_subtask(self, just_exit_freezing_zone=None):
        """Get the current subtask and check if it's completed.

        Args:
            just_exit_freezing_zone: Optional boolean to control VLM calling
                - None or True: Check completion (call VLM)
                - False: Skip VLM call, return current subtask without checking

        Returns:
            str or None: Current subtask or None if all subtasks completed
        """
        if self.current_subtask >= len(self.subtasks):
            return None
        result = self.check_completion(self.subtasks[self.current_subtask], just_exit_freezing_zone)
        if result["result"]["completed"]:
            self.current_subtask += 1
        if self.current_subtask >= len(self.subtasks):
            return None
        return self.subtasks[self.current_subtask]
