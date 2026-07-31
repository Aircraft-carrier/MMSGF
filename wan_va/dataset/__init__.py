# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
"""Dataset package exports for the real-data MOT training path."""

__all__ = [
    'MOT_ACTION_CHUNKS',
    'MOT_ACTION_SEQUENCE_LENGTH',
    'MotBalancedMixDataset',
    'MotGeometryLeRobotData',
    'MotPureLeRobotData',
    'MotTrainData',
    'build_real_mot_base_cache',
    'build_real_mot_training_selection',
    'build_real_mot_train_dataset',
    'load_manifest',
    'mot_action_per_frame',
    'mot_action_sequence_length',
    'mot_latent_frames_per_action_chunk_per_view',
    'mot_sampled_video_frames_per_action_chunk_per_view',
    'quaternion_xyzw_to_matrix',
    'relative_20d_to_absolute_actions',
    'rotation_matrix_to_rot6d_cols',
    'validate_mot_batch_for_forward',
    'validate_mot_geometry_batch',
]


def __getattr__(name):
    if name in {
        'MOT_ACTION_CHUNKS',
        'MOT_ACTION_SEQUENCE_LENGTH',
        'MotBalancedMixDataset',
        'MotGeometryLeRobotData',
        'MotPureLeRobotData',
        'MotTrainData',
        'load_manifest',
        'mot_action_per_frame',
        'mot_action_sequence_length',
        'mot_latent_frames_per_action_chunk_per_view',
        'mot_sampled_video_frames_per_action_chunk_per_view',
        'quaternion_xyzw_to_matrix',
        'relative_20d_to_absolute_actions',
        'rotation_matrix_to_rot6d_cols',
        'validate_mot_batch_for_forward',
        'validate_mot_geometry_batch',
    }:
        from .mot_dataset import (
            MOT_ACTION_CHUNKS,
            MOT_ACTION_SEQUENCE_LENGTH,
            MotBalancedMixDataset,
            MotGeometryLeRobotData,
            MotPureLeRobotData,
            MotTrainData,
            load_manifest,
            mot_action_per_frame,
            mot_action_sequence_length,
            mot_latent_frames_per_action_chunk_per_view,
            mot_sampled_video_frames_per_action_chunk_per_view,
            quaternion_xyzw_to_matrix,
            relative_20d_to_absolute_actions,
            rotation_matrix_to_rot6d_cols,
            validate_mot_batch_for_forward,
            validate_mot_geometry_batch,
        )

        values = {
            'MOT_ACTION_CHUNKS': MOT_ACTION_CHUNKS,
            'MOT_ACTION_SEQUENCE_LENGTH': MOT_ACTION_SEQUENCE_LENGTH,
            'MotBalancedMixDataset': MotBalancedMixDataset,
            'MotGeometryLeRobotData': MotGeometryLeRobotData,
            'MotPureLeRobotData': MotPureLeRobotData,
            'MotTrainData': MotTrainData,
            'load_manifest': load_manifest,
            'mot_action_per_frame': mot_action_per_frame,
            'mot_action_sequence_length': mot_action_sequence_length,
            'mot_latent_frames_per_action_chunk_per_view': mot_latent_frames_per_action_chunk_per_view,
            'mot_sampled_video_frames_per_action_chunk_per_view': mot_sampled_video_frames_per_action_chunk_per_view,
            'quaternion_xyzw_to_matrix': quaternion_xyzw_to_matrix,
            'relative_20d_to_absolute_actions': relative_20d_to_absolute_actions,
            'rotation_matrix_to_rot6d_cols': rotation_matrix_to_rot6d_cols,
            'validate_mot_batch_for_forward': validate_mot_batch_for_forward,
            'validate_mot_geometry_batch': validate_mot_geometry_batch,
        }
        return values[name]
    if name == 'build_real_mot_base_cache':
        from .build_base_valid_data import build_real_mot_base_cache

        return build_real_mot_base_cache
    if name == 'build_real_mot_training_selection':
        from .build_training_selection import build_real_mot_training_selection

        return build_real_mot_training_selection
    if name == 'build_real_mot_train_dataset':
        from .build_training_dataset import build_real_mot_train_dataset

        return build_real_mot_train_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
