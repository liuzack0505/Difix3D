"""Camera-only JSON parser with the metadata interface used by COLMAP Dataset."""

import json

import numpy as np

from .normalize import transform_cameras


def load_target_cameras(path, factor, transform):
    """Read OpenCV camera-to-world poses and convert to the training world."""
    with open(path) as f:
        metadata = json.load(f)
    if not isinstance(metadata, dict):
        raise ValueError("Target camera JSON must be an object.")
    frames = metadata.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("Target camera JSON must contain a non-empty frames list.")
    if factor < 1:
        raise ValueError("data_factor must be positive.")
    poses, intrinsics, sizes = [], [], []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError(f"Target frame {index}: expected an object.")
        camera = {**metadata, **frame}
        if camera.get("camera_model", "OPENCV") not in ("OPENCV", "PINHOLE"):
            raise ValueError(f"Target frame {index}: only OPENCV/PINHOLE is supported.")
        if any(float(camera.get(k, 0)) != 0 for k in ("k1", "k2", "k3", "k4", "p1", "p2")):
            raise ValueError(f"Target frame {index}: supply undistorted cameras (zero distortion).")
        try:
            w, h, fx, fy, cx, cy = [float(camera[k]) for k in ("w", "h", "fl_x", "fl_y", "cx", "cy")]
            pose = np.asarray(frame["transform_matrix"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Target frame {index}: missing or invalid camera parameters.") from exc
        if not np.isfinite([w, h, fx, fy, cx, cy]).all() or min(w, h, fx, fy) <= 0:
            raise ValueError(f"Target frame {index}: invalid dimensions or intrinsics.")
        width, height = int(w) // factor, int(h) // factor
        if width < 1 or height < 1 or w != int(w) or h != int(h):
            raise ValueError(f"Target frame {index}: invalid image size after downsampling.")
        if pose.shape != (4, 4) or not np.isfinite(pose).all() or not np.allclose(pose[3], [0, 0, 0, 1]):
            raise ValueError(f"Target frame {index}: expected a finite homogeneous 4x4 matrix.")
        if not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4) or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4):
            raise ValueError(f"Target frame {index}: camera rotation must be orthonormal.")
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        K[:2] /= factor
        poses.append(pose)
        intrinsics.append(K)
        sizes.append((width, height))
    return transform_cameras(transform, np.stack(poses)).astype(np.float32), intrinsics, sizes


class TargetParser:
    """Parse target views without requiring images or a COLMAP reconstruction.

    test_every=0 disables splitting, so Dataset(..., split="test") includes
    every target frame. The same metadata can later describe corrected images.
    """

    def __init__(self, path, factor=1, transform=None):
        self.factor = factor
        self.test_every = 0
        self.transform = np.eye(4) if transform is None else np.asarray(transform)
        self.camtoworlds, intrinsics, sizes = load_target_cameras(
            path, factor, self.transform
        )
        self.camera_ids = list(range(len(self.camtoworlds)))
        self.Ks_dict = dict(enumerate(intrinsics))
        self.imsize_dict = dict(enumerate(sizes))
        self.params_dict = {i: np.empty(0, dtype=np.float32) for i in self.camera_ids}
        self.mask_dict = {i: None for i in self.camera_ids}
        self.image_names = [f"{i:04d}.png" for i in self.camera_ids]
        self.image_paths = [None] * len(self.camera_ids)
        self.alpha_mask_paths = None
