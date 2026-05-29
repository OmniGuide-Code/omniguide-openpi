import numpy as np
import requests

from collision.vggt_pcd import preprocess_images


class VLMPointer:
    def __init__(self, base_url, camera_view="right"):
        self.base_url = base_url
        self.camera_view = camera_view

    def get_3d_pointer(self, observation, task, subtask):
        return self.request(observation, task, subtask)

    def request(self, observation, task, subtask):
        camera_image_map = {
            "left": observation.left_image,
            "right": observation.right_image,
            "overhead": observation.overhead_image,
        }
        img_tensor = camera_image_map.get(self.camera_view, observation.right_image)
        # Preprocess to match the resolution VGGT uses for point_map (width=518, crop mode).
        # The VLM must see the same image so its returned pixel coordinates index correctly
        # into point_map without requiring any coordinate rescaling.
        img_tensor = preprocess_images(img_tensor)  # (1, C, H', W') same space as point_map
        img_np = img_tensor[0].permute(1, 2, 0).cpu().numpy()
        img_uint8 = (img_np * 255).astype(np.uint8)
        payload = {
            "task_description": task,
            "subtask": subtask,
            "img": img_uint8.tolist(),
        }

        r = requests.post(f"{self.base_url}/pointing", json=payload, timeout=120)
        print("predict result:", r.json())

        # VLM returns [y_px, x_px] = [row, col] in the preprocessed image's coordinate space.
        r = r.json()["result"]["final_points"][0]["point"]
        return r  # [row, col], both valid within the preprocessed (H×W) image
