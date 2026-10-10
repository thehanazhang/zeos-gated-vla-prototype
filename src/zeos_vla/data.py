"""LeRobot v3 adapter with episode-safe future chunks and explicit padding masks."""

import torch
from torch.utils.data import Dataset
from PIL import Image


class LeRobotAdapter(Dataset):
    def __init__(
        self,
        repo_id,
        revision,
        episodes,
        camera_key="observation.images.wrist",
        instruction=None,
        root=None,
    ):
        from lerobot.datasets.lerobot_dataset import (
            LeRobotDataset,
            LeRobotDatasetMetadata,
        )

        if not revision or not episodes or len(set(episodes)) != len(episodes):
            raise ValueError(
                "Pinned revision and unique explicit episode selection required"
            )
        meta = LeRobotDatasetMetadata(repo_id, root=root, revision=revision)
        self.raw = LeRobotDataset(
            repo_id,
            root=root,
            revision=revision,
            episodes=episodes,
            delta_timestamps={"action": [i / meta.fps for i in range(16)]},
            video_backend="pyav",
        )
        self.camera_key = camera_key
        self.instruction = instruction
        self.revision = revision
        self.raw._ensure_hf_dataset_loaded()
        self.episode_ids = torch.as_tensor(
            list(self.raw.hf_dataset["episode_index"])
        ).flatten()
        self.indices = list(range(len(self.raw)))

    def select_frames(self, frames_per_episode):
        if frames_per_episode < 1:
            raise ValueError("frames_per_episode must be positive")
        indices = []
        for ep in self.episode_ids.unique().tolist():
            candidates = (self.episode_ids == ep).nonzero().flatten()
            offsets = (
                torch.linspace(
                    0, len(candidates) - 1, min(frames_per_episode, len(candidates))
                )
                .round()
                .long()
            )
            indices.extend(candidates[offsets].tolist())
        self.indices = indices
        return self

    def __len__(self):
        return len(self.indices)

    @staticmethod
    def adapt(item, camera_key, instruction=None):
        pixels = item[camera_key]
        if pixels.ndim != 3 or pixels.shape[0] != 3:
            raise ValueError("Expected LeRobot camera CHW RGB tensor")
        image = Image.fromarray(
            (pixels.clamp(0, 1).permute(1, 2, 0) * 255).round().byte().numpy()
        )
        actions = torch.as_tensor(item["action"]).float()
        state = torch.as_tensor(item["observation.state"]).float()
        pad = torch.as_tensor(item["action_is_pad"]).bool()
        if actions.ndim != 2 or actions.shape[0] != 16 or pad.shape != (16,):
            raise ValueError("Expected a 16-step action chunk and LeRobot padding mask")
        if not torch.isfinite(actions).all() or not torch.isfinite(state).all():
            raise ValueError("Dataset contains nonfinite values")
        return {
            "image": image,
            "instruction": instruction or item["task"],
            "actions": actions,
            "proprio": state,
            "action_mask": ~pad,
            "episode_index": int(item["episode_index"]),
            "frame_index": int(item["frame_index"]),
        }

    def __getitem__(self, index):
        return self.adapt(
            self.raw[self.indices[index]], self.camera_key, self.instruction
        )
