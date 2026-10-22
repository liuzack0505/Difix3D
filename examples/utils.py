import numpy as np
import torch
from scipy.spatial.transform import Rotation


class CameraPoseInterpolator:
    """
    A system for interpolating between sets of camera poses with visualization capabilities.
    """
    
    def __init__(self, rotation_weight=1.0, translation_weight=1.0):
        """
        Initialize the interpolator with weights for pose distance computation.
        
        Args:
            rotation_weight: Weight for rotational distance in pose matching
            translation_weight: Weight for translational distance in pose matching
        """
        self.rotation_weight = rotation_weight
        self.translation_weight = translation_weight
    
    def compute_pose_distance(self, pose1, pose2):
        """
        Compute weighted distance between two camera poses.
        
        Args:
            pose1, pose2: 4x4 transformation matrices
            
        Returns:
            Combined weighted distance between poses
        """
        # Translation distance (Euclidean)
        t1, t2 = pose1[:3, 3], pose2[:3, 3]
        translation_dist = np.linalg.norm(t1 - t2)
        
        # Rotation distance (angular distance between quaternions)
        R1 = Rotation.from_matrix(pose1[:3, :3])
        R2 = Rotation.from_matrix(pose2[:3, :3])
        q1 = R1.as_quat()
        q2 = R2.as_quat()
        
        # Ensure quaternions are in the same hemisphere
        if np.dot(q1, q2) < 0:
            q2 = -q2
        
        rotation_dist = np.arccos(np.clip(2 * np.dot(q1, q2)**2 - 1, -1.0, 1.0))
        
        return (self.translation_weight * translation_dist +
                self.rotation_weight * rotation_dist)

    def compute_normalized_pose_distance(self, pose1, pose2, scene_poses, translation_lambda=1.0):
        """
        Compute the scale-normalized camera pose distance
            d_ij = θ_ij / π + λ_t ||t_i - t_j||_2 / r̄,
        where θ_ij ∈ [0, π] is the SO(3) geodesic angle between R_i and R_j and
        r̄ = (1/N) Σ_k ||t_k||_2 is the mean L2 norm of the scene's camera positions. The rotation
        term lies in [0, 1] and translations are measured in units of the mean radius, so both
        terms have the same order of magnitude in any scene scale.

        Args:
            pose1, pose2: 4x4 camera-to-world transformation matrices
            scene_poses: [N, 4, 4] array of the scene's camera-to-world poses, used to compute r̄
            translation_lambda: λ_t, weight of the translation term

        Returns:
            Normalized distance between the poses
        """
        mean_radius = np.linalg.norm(np.asarray(scene_poses)[:, :3, 3], axis=-1).mean()
        if mean_radius <= 0:
            raise ValueError("Mean camera radius must be positive; all cameras are at the origin.")

        # Geodesic angle: magnitude of the relative rotation R_i^T R_j, in [0, π]
        relative = Rotation.from_matrix(pose1[:3, :3]).inv() * Rotation.from_matrix(pose2[:3, :3])
        rotation_dist = relative.magnitude() / np.pi

        translation_dist = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3]) / mean_radius

        return rotation_dist + translation_lambda * translation_dist

    def find_nearest_assignments(self, training_poses, testing_poses):
        """
        Find the nearest training camera pose for each testing camera pose.
        
        Args:
            training_poses: [N, 4, 4] array of training camera poses
            testing_poses: [M, 4, 4] array of testing camera poses
            
        Returns:
            assignments: list of closest training pose indices for each testing pose
        """
        M = len(testing_poses)
        assignments = []

        for j in range(M):
            # Compute distance from each training pose to this testing pose
            distances = [self.compute_pose_distance(training_pose, testing_poses[j])
                         for training_pose in training_poses]
            # Find the index of the nearest training pose
            nearest_index = np.argmin(distances)
            assignments.append(nearest_index)
        
        return assignments

    def find_nearest_normalized_assignments(self, training_poses, testing_poses, translation_lambda=1.0):
        """
        Find the nearest training camera pose for each testing camera pose using
        compute_normalized_pose_distance, with the mean camera radius taken from the training poses.

        Args:
            training_poses: [N, 4, 4] array of training camera poses
            testing_poses: [M, 4, 4] array of testing camera poses
            translation_lambda: λ_t, weight of the translation term

        Returns:
            assignments: list of closest training pose indices for each testing pose
        """
        assignments = []
        for testing_pose in testing_poses:
            distances = [self.compute_normalized_pose_distance(training_pose, testing_pose, training_poses, translation_lambda)
                         for training_pose in training_poses]
            assignments.append(int(np.argmin(distances)))
        return assignments

    @torch.no_grad()
    def find_most_overlay_gaussian_assignments(self, training_poses, testing_poses, means, render_depth,
                                               training_Ks, training_sizes, testing_Ks, testing_sizes,
                                               downscale=4, depth_tolerance=0.05, chunk_size=1 << 18):
        """
        Find the training camera that sees the most Gaussians in common with each testing camera.

        A Gaussian is visible in a camera when its center projects inside the image, lies in front of
        the camera, and is not occluded: its depth is at most (1 + depth_tolerance) times the rendered
        depth at that pixel.

        Args:
            training_poses: [N, 4, 4] array of training camera-to-world poses
            testing_poses: [M, 4, 4] array of testing camera-to-world poses
            means: [G, 3] tensor of Gaussian centers; its device is used for all computation
            render_depth: callable (camtoworld [4, 4], K [3, 3], width, height) -> [height, width] depth
                tensor, with inf where nothing is rendered
            training_Ks, testing_Ks: [N or M, 3, 3] intrinsics at full resolution
            training_sizes, testing_sizes: lists of (width, height) at full resolution
            downscale: resolution divisor for the depth maps used in the occlusion test
            depth_tolerance: relative depth slack for the occlusion test
            chunk_size: number of Gaussians processed at once

        Returns:
            assignments: list of training pose indices sharing the most visible Gaussians with each testing pose
        """
        device = means.device

        def prepare_cameras(poses, Ks, sizes):
            camtoworlds = torch.as_tensor(np.asarray(poses), dtype=torch.float32, device=device)
            Ks = torch.as_tensor(np.asarray(Ks), dtype=torch.float32, device=device).clone()
            Ks[:, :2] /= downscale
            sizes = [(max(w // downscale, 1), max(h // downscale, 1)) for w, h in sizes]
            depths = [render_depth(c2w, K, w, h) for c2w, K, (w, h) in zip(camtoworlds, Ks, sizes)]
            return torch.linalg.inv(camtoworlds), Ks, sizes, depths

        def visibility(cameras, points):
            viewmats, Ks, sizes, depths = cameras
            visible = torch.zeros(len(sizes), len(points), device=device)
            for c, (w, h) in enumerate(sizes):
                points_cam = points @ viewmats[c, :3, :3].T + viewmats[c, :3, 3]
                z = points_cam[:, 2]
                uv = points_cam[:, :2] / z.clamp(min=1e-6)[:, None] @ Ks[c, :2, :2].T + Ks[c, :2, 2]
                u, v = uv[:, 0].floor().long(), uv[:, 1].floor().long()
                inside = (z > 1e-6) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
                surface = depths[c][v.clamp(0, h - 1), u.clamp(0, w - 1)]
                visible[c] = (inside & (z <= surface * (1 + depth_tolerance))).float()
            return visible

        training_cameras = prepare_cameras(training_poses, training_Ks, training_sizes)
        testing_cameras = prepare_cameras(testing_poses, testing_Ks, testing_sizes)

        # overlap[i, j]: number of Gaussians visible in both training camera i and testing camera j
        overlap = torch.zeros(len(training_poses), len(testing_poses), device=device)
        for start in range(0, len(means), chunk_size):
            points = means[start:start + chunk_size].float()
            overlap += visibility(training_cameras, points) @ visibility(testing_cameras, points).T

        return overlap.argmax(dim=0).tolist()

    def interpolate_rotation(self, R1, R2, t):
        """
        Interpolate between two rotation matrices using SLERP.
        """
        q1 = Rotation.from_matrix(R1).as_quat()
        q2 = Rotation.from_matrix(R2).as_quat()
        
        if np.dot(q1, q2) < 0:
            q2 = -q2
        
        # Clamp dot product to avoid invalid values in arccos
        dot_product = np.clip(np.dot(q1, q2), -1.0, 1.0)
        theta = np.arccos(dot_product)
        
        if np.abs(theta) < 1e-6:
            q_interp = (1 - t) * q1 + t * q2
        else:
            q_interp = (np.sin((1-t)*theta) * q1 + np.sin(t*theta) * q2) / np.sin(theta)
        
        q_interp = q_interp / np.linalg.norm(q_interp)
        return Rotation.from_quat(q_interp).as_matrix()
    
    def interpolate_poses(self, training_poses, testing_poses, num_steps=20):
        """
        Interpolate between camera poses using nearest assignments.
        
        Args:
            training_poses: [N, 4, 4] array of training poses
            testing_poses: [M, 4, 4] array of testing poses
            num_steps: number of interpolation steps
            
        Returns:
            interpolated_sequences: list of lists of interpolated poses
        """
        assignments = self.find_nearest_assignments(training_poses, testing_poses)
        interpolated_sequences = []
        
        for test_idx, train_idx in enumerate(assignments):
            train_pose = training_poses[train_idx]
            test_pose = testing_poses[test_idx]
            sequence = []
            
            for t in np.linspace(0, 1, num_steps):
                # Interpolate rotation
                R_interp = self.interpolate_rotation(
                    train_pose[:3, :3],
                    test_pose[:3, :3],
                    t
                )
                
                # Interpolate translation
                t_interp = (1-t) * train_pose[:3, 3] + t * test_pose[:3, 3]
                
                # Construct interpolated pose
                pose_interp = np.eye(4)
                pose_interp[:3, :3] = R_interp
                pose_interp[:3, 3] = t_interp
                
                sequence.append(pose_interp)
            
            interpolated_sequences.append(sequence)
        
        return interpolated_sequences


    def shift_poses(self, training_poses, testing_poses, distance=0.1, threshold=0.1):
        """
        Shift nearest training poses toward testing poses by a specified distance.
        
        Args:
            training_poses: [N, 4, 4] array of training camera poses
            testing_poses: [M, 4, 4] array of testing camera poses
            distance: float, the step size to move training pose toward testing pose
            
        Returns:
            novel_poses: [M, 4, 4] array of shifted poses
        """
        assignments = self.find_nearest_assignments(training_poses, testing_poses)
        novel_poses = []

        for test_idx, train_idx in enumerate(assignments):
            train_pose = training_poses[train_idx]
            test_pose = testing_poses[test_idx]

            if self.compute_pose_distance(train_pose, test_pose) <= distance:
                novel_poses.append(test_pose)
                continue

            # Calculate translation step if shifting is necessary
            t1, t2 = train_pose[:3, 3], test_pose[:3, 3]
            translation_direction = t2 - t1
            translation_norm = np.linalg.norm(translation_direction)
            
            if translation_norm > 1e-6:
                translation_step = (translation_direction / translation_norm) * distance
                new_translation = t1 + translation_step
            else:
                # If translation direction is too small, use testing pose translation directly
                new_translation = t2

            # Check if the new translation would overshoot the testing pose translation
            if np.dot(new_translation - t1, t2 - t1) <= 0 or np.linalg.norm(new_translation - t2) <= distance:
                new_translation = t2

            # Update rotation
            R1 = train_pose[:3, :3]
            R2 = test_pose[:3, :3]
            if translation_norm > 1e-6:
                R_interp = self.interpolate_rotation(R1, R2, min(distance / translation_norm, 1.0))
            else:
                R_interp = R2  # Use testing rotation if too close

            # Construct shifted pose
            shifted_pose = np.eye(4)
            shifted_pose[:3, :3] = R_interp
            shifted_pose[:3, 3] = new_translation

            novel_poses.append(shifted_pose)

        return np.array(novel_poses)