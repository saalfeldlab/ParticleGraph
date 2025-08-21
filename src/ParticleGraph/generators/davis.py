import logging
import os
import cv2
from contextlib import contextmanager
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as nnf
from datamate import Directory, Namespace, root
from tqdm import tqdm

from flyvis import renderings_dir

from flyvis.datasets.augmentation.hex import (
    ContrastBrightness,
    GammaCorrection,
    HexFlip,
    HexRotate,
    PixelNoise,
)
from flyvis.datasets.augmentation.temporal import (
    CropFrames,
    Interpolate,
)
from flyvis.datasets import MultiTaskDataset
from flyvis.datasets.rendering import BoxEye
from flyvis.datasets.rendering.utils import split

logger = logging.getLogger(__name__)

__all__ = ["RenderedDavis", "MultiTaskDavis", "AugmentedDavis"]


# ============================================================================
# DAVIS Utility Functions (equivalent to sintel_utils)
# FROM WEBSITE # https://davischallenge.org/
# ============================================================================

def load_image_sequence(sequence_dir: Path, start_frame: int = 0, end_frame: Optional[int] = None) -> np.ndarray:
    """Load frames from a directory of numbered JPEG images.

    Args:
        sequence_dir: Directory containing numbered JPEG files (00000.jpg, 00001.jpg, etc.)
        start_frame: First frame to load
        end_frame: Last frame to load (None = all frames)

    Returns:
        Array of shape (frames, height, width, 3) with values in [0, 1]
    """
    # Find all JPEG files
    jpeg_files = sorted([f for f in sequence_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg']])

    if not jpeg_files:
        raise ValueError(f"No JPEG files found in {sequence_dir}")

    # Set frame range
    total_frames = len(jpeg_files)
    if end_frame is None:
        end_frame = total_frames
    end_frame = min(end_frame, total_frames)

    frames = []
    for i in range(start_frame, end_frame):
        if i < len(jpeg_files):
            # Load image
            img = cv2.imread(str(jpeg_files[i]))
            if img is None:
                logger.warning(f"Could not load image: {jpeg_files[i]}")
                continue

            # Convert BGR to RGB and normalize to [0, 1]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_normalized = img_rgb.astype(np.float32) / 255.0
            frames.append(img_normalized)

    if not frames:
        raise ValueError(f"No valid frames loaded from {sequence_dir}")

    return np.array(frames)  # (frames, height, width, 3)


def sample_lum_from_frame(frame: np.ndarray) -> np.ndarray:
    """Convert RGB frame to grayscale luminance.

    Args:
        frame: RGB frame of shape (height, width, 3) or (3, height, width)

    Returns:
        Grayscale image of shape (height, width)
    """
    if frame.shape[0] == 3:  # (3, H, W) format
        frame = frame.transpose(1, 2, 0)  # -> (H, W, 3)

    # Convert to grayscale using standard weights
    gray = 0.299 * frame[:, :, 0] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 2]
    return gray


def davis_meta(rendered_davis, davis_path: Path, n_frames: int, vertical_splits: int) -> Namespace:
    """Create metadata object for DAVIS dataset.

    Args:
        rendered_davis: RenderedDavis object
        davis_path: Root directory containing sequence directories
        n_frames: Number of frames per sequence
        vertical_splits: Number of vertical splits

    Returns:
        Metadata namespace
    """
    # Find all sequence directories (each contains numbered JPEG files)
    sequence_dirs = sorted([d for d in davis_path.iterdir() if d.is_dir()])

    # Get sequence properties
    sequence_indices = []
    frames_per_scene = []
    sequence_index_to_splits = {}

    for i, seq_dir in enumerate(sequence_dirs):
        # Count JPEG files in directory
        jpeg_files = [f for f in seq_dir.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg']]
        if jpeg_files:
            total_frames = len(jpeg_files)
            sequence_indices.append(i)
            frames_per_scene.append(total_frames)

            # Map sequence index to its splits
            split_indices = []
            for j in range(vertical_splits):
                split_idx = i * vertical_splits + j
                split_indices.append(split_idx)
            sequence_index_to_splits[i] = split_indices

    return Namespace(
        sequence_paths=sequence_dirs,
        sequence_indices=np.array(sequence_indices),
        frames_per_scene=np.array(frames_per_scene),
        sequence_index_to_splits=sequence_index_to_splits,
    )


def temporal_split_cached_samples(cached_sequences: List[Dict], n_frames: int, split: bool = True) -> Tuple[
    List[Dict], np.ndarray]:
    """Split sequences temporally into chunks of n_frames.

    Args:
        cached_sequences: List of sequence dictionaries
        n_frames: Target frames per chunk
        split: Whether to perform temporal splitting

    Returns:
        Tuple of (split_sequences, original_repeats)
    """
    if not split:
        return cached_sequences, np.ones(len(cached_sequences), dtype=int)

    split_sequences = []
    original_repeats = []

    for seq in cached_sequences:
        lum = seq["lum"]  # (frames, channels, hexals)
        total_frames = lum.shape[0]

        if total_frames <= n_frames:
            split_sequences.append(seq)
            original_repeats.append(1)
        else:
            # Split into overlapping chunks
            n_chunks = max(1, (total_frames - n_frames) // (n_frames // 2) + 1)
            chunk_starts = np.linspace(0, total_frames - n_frames, n_chunks).astype(int)

            for start in chunk_starts:
                chunk_seq = {}
                for key, value in seq.items():
                    chunk_seq[key] = value[start:start + n_frames]
                split_sequences.append(chunk_seq)

            original_repeats.append(len(chunk_starts))

    return split_sequences, np.array(original_repeats)


def original_train_and_validation_indices(dataset) -> Tuple[List[int], List[int]]:
    """Get train/validation split for DAVIS dataset.

    Args:
        dataset: MultiTaskDavis dataset object

    Returns:
        Tuple of (train_indices, validation_indices)
    """
    total_sequences = len(dataset)
    train_ratio = 0.8

    # Simple split - first 80% for training, rest for validation
    split_point = int(total_sequences * train_ratio)

    train_indices = list(range(split_point))
    val_indices = list(range(split_point, total_sequences))

    return train_indices, val_indices


# ============================================================================
# Main Dataset Classes
# ============================================================================

@root(renderings_dir)
class RenderedDavis(Directory):
    """Rendering and referencing rendered DAVIS data.

    Args:
        tasks: List of tasks to include in the rendering. Only 'lum' supported for videos.
        boxfilter: Key word arguments for the BoxEye filter.
        vertical_splits: Number of vertical splits of each frame.
        n_frames: Minimum number of frames required for a sequence to be used.
        max_frames: Maximum number of frames to render per sequence. If None, use all available frames.
        center_crop_fraction: Fraction of the image to keep after cropping.
        unittest: If True, only renders a single sequence.
        davis_path: Path to the directory containing DAVIS video sequence folders (each folder = one sequence).

    Attributes:
        config: Configuration parameters used for rendering.
        sequence_<id>_<name>_split_<j>/lum (ArrayFile):
            Rendered luminance data (frames, 1, hexals).
    """

    def __init__(
            self,
            tasks: List[str] = ["lum"],
            boxfilter: Dict[str, int] = dict(extent=15, kernel_size=13),
            vertical_splits: int = 3,
            n_frames: int = 19,
            max_frames: Optional[int] = None,
            center_crop_fraction: float = 0.7,
            unittest: bool = False,
            davis_path: Optional[Union[str, Path]] = None,
    ):
        if davis_path is None:
            raise ValueError("davis_path must be provided - path to directory containing video files")

        root_dir = Path(davis_path)
        if not root_dir.exists():
            raise ValueError(f"Directory does not exist: {root_dir}")

        # Only luminance is available from image sequences (no flow/depth)
        if "flow" in tasks or "depth" in tasks:
            logger.warning("Flow and depth not available for DAVIS image sequences, using only 'lum'")
        tasks = ["lum"]

        boxfilter = BoxEye(**boxfilter)

        # Find all sequence directories (DAVIS format: each sequence is a directory of JPEGs)
        sequence_dirs = [d for d in root_dir.iterdir() if d.is_dir()]

        if not sequence_dirs:
            raise ValueError(f"No sequence directories found in {root_dir}")

        logger.info(f"Found {len(sequence_dirs)} sequence directories")

        for i, seq_dir in enumerate(tqdm(sequence_dirs, desc="Rendering sequences")):
            try:
                # Load full image sequence
                frames = load_image_sequence(seq_dir, end_frame=None)

                # Skip if sequence too short
                if len(frames) < n_frames:
                    logger.warning(f"Sequence {seq_dir.name} has only {len(frames)} frames, skipping")
                    continue

                # Apply max_frames cap if requested
                if max_frames is not None and len(frames) > max_frames:
                    frames = frames[:max_frames]

                # Convert to grayscale luminance
                lum_frames = []
                for frame in frames:
                    gray = sample_lum_from_frame(frame)
                    gray = np.rot90(gray, k=-1)
                    lum_frames.append(gray)

                lum = np.array(lum_frames)  # (frames, height, width)

                # Apply spatial splitting and hexagonal sampling
                lum_split = split(
                    lum,
                    boxfilter.min_frame_size[1] + 2 * boxfilter.kernel_size,
                    vertical_splits,
                    center_crop_fraction,
                )

                # Apply hexagonal sampling: (splits, frames, 1, #hexals)
                lum_hex = boxfilter(lum_split).cpu()

                # Store each split
                for j in range(lum_hex.shape[0]):
                    sequence_name = seq_dir.name
                    path = f"sequence_{i:02d}_{sequence_name}_split_{j:02d}"
                    self[f"{path}/lum"] = lum_hex[j]

            except Exception as e:
                logger.error(f"Error processing sequence {seq_dir.name}: {e}")
                continue

            if unittest:
                break

    def __call__(self, seq_id: int) -> Dict[str, np.ndarray]:
        """Returns all rendered data for a given sequence index.

        Args:
            seq_id: Index of the sequence to retrieve.

        Returns:
            Dictionary containing the rendered data for the specified sequence.
        """
        data = self[sorted(self)[seq_id]]
        return {key: data[key][:] for key in sorted(data)}



class MultiTaskDavis(MultiTaskDataset):
    """DAVIS image sequence dataset.

    Args:
        root_dir: Directory containing DAVIS sequence directories (each sequence is a folder of JPEGs).
        tasks: List of tasks to include. Only 'lum' supported for image sequences.
        boxfilter: Key word arguments for the BoxEye filter.
        vertical_splits: Number of vertical splits of each frame.
        n_frames: Number of frames to render for each sequence.
        center_crop_fraction: Fraction of the image to keep after cropping.
        dt: Sampling and integration time constant.
        augment: Turns augmentation on and off.
        random_temporal_crop: Randomly crops a temporal window of length `n_frames` from
            each sequence.
        all_frames: If True, all frames are returned. If False, only `n_frames`. Takes
            precedence over `random_temporal_crop`.
        resampling: If True, piecewise-constant resamples the input sequence to the
            target framerate (1/dt).
        interpolate: If True, linearly interpolates the target sequence to the target
            framerate (1/dt).
        p_flip: Probability of flipping the sequence across hexagonal axes.
        p_rot: Probability of rotating the sequence by n*60 degrees.
        contrast_std: Standard deviation of the contrast augmentation.
        brightness_std: Standard deviation of the brightness augmentation.
        gaussian_white_noise: Standard deviation of the pixel-wise gaussian white noise.
        gamma_std: Standard deviation of the gamma augmentation.
        _init_cache: If True, caches the dataset in memory.
        unittest: If True, only renders a single sequence.
        flip_axes: List of axes to flip over.

    Attributes:
        dt (float): Sampling and integration time constant.
        t_pre (float): Warmup time.
        t_post (float): Cooldown time.
        tasks (List[str]): List of all tasks.
        valid_tasks (List[str]): List of valid task names.

    Raises:
        ValueError: If any element in tasks is invalid.
    """

    original_framerate: int = 24  # DAVIS sequences are typically around this framerate
    dt: float = 1 / 50
    t_pre: float = 0.0
    t_post: float = 0.0
    tasks: List[str] = []
    valid_tasks: List[str] = ["lum"]  # Only luminance available for image sequences

    def __init__(
            self,
            root_dir: Union[str, Path],
            tasks: List[str] = ["lum"],
            boxfilter: Dict[str, int] = dict(extent=15, kernel_size=13),
            vertical_splits: int = 3,
            n_frames: int = 19,
            center_crop_fraction: float = 0.7,
            dt: float = 1 / 50,
            augment: bool = True,
            random_temporal_crop: bool = True,
            all_frames: bool = False,
            resampling: bool = True,
            interpolate: bool = True,
            p_flip: float = 0.5,
            p_rot: float = 5 / 6,
            contrast_std: float = 0.2,
            brightness_std: float = 0.1,
            gaussian_white_noise: float = 0.08,
            gamma_std: Optional[float] = None,
            _init_cache: bool = True,
            unittest: bool = False,
            flip_axes: List[int] = [0, 1],
    ):
        def check_tasks(tasks):
            invalid_tasks = [x for x in tasks if x not in self.valid_tasks]
            if invalid_tasks:
                raise ValueError(f"invalid tasks {invalid_tasks}")

            tasks = [v for v in self.valid_tasks if v in tasks]  # sort
            # because the input 'lum' is always required
            data_keys = tasks if "lum" in tasks else ["lum", *tasks]
            return tasks, data_keys

        self.tasks, self.data_keys = check_tasks(tasks)
        self.interpolate = interpolate
        self.n_frames = n_frames if not unittest else 3
        self.dt = dt

        self.all_frames = all_frames
        self.resampling = resampling

        self.boxfilter = boxfilter
        self.extent = boxfilter["extent"]
        assert vertical_splits >= 1
        self.vertical_splits = vertical_splits
        self.center_crop_fraction = center_crop_fraction

        self.p_flip = p_flip
        self.p_rot = p_rot
        self.contrast_std = contrast_std
        self.brightness_std = brightness_std
        self.gaussian_white_noise = gaussian_white_noise
        self.gamma_std = gamma_std
        self.random_temporal_crop = random_temporal_crop
        self.flip_axes = flip_axes
        self.fix_augmentation_params = False

        self.init_augmentation()
        self._augmentations_are_initialized = True
        # note: self.augment is a property with a setter that relies on
        # _augmentations_are_initialized
        self.augment = augment

        self.unittest = unittest
        self.root_dir = Path(root_dir)

        self.rendered = RenderedDavis(
            tasks=tasks,
            boxfilter=boxfilter,
            vertical_splits=vertical_splits,
            n_frames=n_frames,
            center_crop_fraction=center_crop_fraction,
            unittest=unittest,
            davis_path=root_dir,
        )

        self.meta = davis_meta(
            self.rendered, self.root_dir, n_frames, vertical_splits
        )

        self.config = Namespace(
            root_dir=str(root_dir),
            tasks=tasks,
            interpolate=interpolate,
            n_frames=n_frames,
            dt=dt,
            augment=augment,
            all_frames=all_frames,
            resampling=resampling,
            random_temporal_crop=random_temporal_crop,
            boxfilter=boxfilter,
            vertical_splits=vertical_splits,
            p_flip=p_flip,
            p_rot=p_rot,
            contrast_std=contrast_std,
            brightness_std=brightness_std,
            gaussian_white_noise=gaussian_white_noise,
            gamma_std=gamma_std,
            center_crop_fraction=center_crop_fraction,
            flip_axes=flip_axes,
        )

        # Create metadata - handle case where some sequences were skipped
        n_rendered_items = len(self.rendered)

        if n_rendered_items == 0:
            raise ValueError(f"No sequences were successfully rendered. Try reducing n_frames (currently {n_frames})")

        # Create arrays of the right length for successfully rendered sequences
        # Note: len(self.rendered) gives the number of rendered items (sequences * splits)
        self.arg_df = pd.DataFrame(
            dict(
                index=np.arange(n_rendered_items),
                original_index=np.tile(np.arange(n_rendered_items // vertical_splits), vertical_splits),
                name=sorted(self.rendered.keys()),
                original_n_frames=np.full(n_rendered_items, n_frames),  # All rendered sequences have n_frames
            )
        )

        if _init_cache:
            self.init_cache()

    def init_cache(self) -> None:
        """Initialize the cache with preprocessed sequences."""
        self.cached_sequences = [
            {
                key: torch.tensor(val, dtype=torch.float32)
                for key, val in self.rendered(seq_id).items()
                if key in self.data_keys
            }
            for seq_id in range(len(self))
        ]

    def __repr__(self) -> str:
        repr = f"{self.__class__.__name__} with {len(self)} sequences.\n"
        repr += "See docs, arg_df and meta for more details.\n"
        return repr

    @property
    def docs(self) -> str:
        print(self.__doc__)

    def __setattr__(self, name: str, value: Any) -> None:
        """Custom attribute setter to handle special cases and update augmentation.

        Args:
            name: Name of the attribute to set.
            value: Value to set the attribute to.

        Raises:
            AttributeError: If trying to change framerate or rendered initialization
                attributes.
        """
        # some changes have no effect cause they are fixed, or set by the pre-rendering
        if name == "framerate":
            raise AttributeError("cannot change framerate")
        if hasattr(self, "rendered") and name in self.rendered.config:
            raise AttributeError("cannot change attribute of rendered initialization")
        super().__setattr__(name, value)
        # also update augmentation because it may already be initialized
        if getattr(self, "_augmentations_are_initialized", False):
            self.update_augmentation(name, value)

    def init_augmentation(self) -> None:
        """Initialize augmentation callables."""
        self.temporal_crop = CropFrames(
            self.n_frames, all_frames=self.all_frames, random=self.random_temporal_crop
        )
        self.jitter = ContrastBrightness(
            contrast_std=self.contrast_std, brightness_std=self.brightness_std
        )
        self.rotate = HexRotate(self.extent, p_rot=self.p_rot)
        self.flip = HexFlip(self.extent, p_flip=self.p_flip, flip_axes=self.flip_axes)
        self.noise = PixelNoise(self.gaussian_white_noise)

        self.piecewise_resample = Interpolate(
            self.original_framerate, 1 / self.dt, mode="nearest-exact"
        )
        self.linear_interpolate = Interpolate(
            self.original_framerate,
            1 / self.dt,
            mode="linear",
        )
        self.gamma_correct = GammaCorrection(1, self.gamma_std)

    def update_augmentation(self, name: str, value: Any) -> None:
        """Update augmentation parameters based on attribute changes.

        Args:
            name: Name of the attribute that changed.
            value: New value of the attribute.
        """
        if name == "dt":
            self.piecewise_resample.target_framerate = 1 / value
            self.linear_interpolate.target_framerate = 1 / value
        if name in ["all_frames", "random_temporal_crop"]:
            self.temporal_crop.all_frames = value
            self.temporal_crop.random = value
        if name in ["contrast_std", "brightness_std"]:
            self.jitter.contrast_std = value
            self.jitter.brightness_std = value
        if name == "p_rot":
            self.rotate.p_rot = value
        if name == "p_flip":
            self.flip.p_flip = value
        if name == "gaussian_white_noise":
            self.noise.std = value
        if name == "gamma_std":
            self.gamma_correct.std = value

    def set_augmentation_params(
            self,
            n_rot: Optional[int] = None,
            flip_axis: Optional[int] = None,
            contrast_factor: Optional[float] = None,
            brightness_factor: Optional[float] = None,
            gaussian_white_noise: Optional[float] = None,
            gamma: Optional[float] = None,
            start_frame: Optional[int] = None,
            total_sequence_length: Optional[int] = None,
    ) -> None:
        """Set augmentation callable parameters.

        Info:
            Called for each call of get_item.

        Args:
            n_rot: Number of rotations to apply.
            flip_axis: Axis to flip over.
            contrast_factor: Contrast factor for jitter augmentation.
            brightness_factor: Brightness factor for jitter augmentation.
            gaussian_white_noise: Standard deviation for noise augmentation.
            gamma: Gamma value for gamma correction.
            start_frame: Starting frame for temporal crop.
            total_sequence_length: Total length of the sequence.
        """
        if not self.fix_augmentation_params:
            self.rotate.set_or_sample(n_rot)
            self.flip.set_or_sample(flip_axis)
            self.jitter.set_or_sample(contrast_factor, brightness_factor)
            self.noise.set_or_sample(gaussian_white_noise)
            self.gamma_correct.set_or_sample(gamma)
            self.temporal_crop.set_or_sample(
                start=start_frame, total_sequence_length=total_sequence_length
            )

    def get_item(self, key: int) -> Dict[str, torch.Tensor]:
        """Return a dataset sample.

        Args:
            key: Index of the sample to retrieve.

        Returns:
            Dictionary containing the augmented sample data.
        """
        return self.apply_augmentation(self.cached_sequences[key])

    @contextmanager
    def augmentation(self, abool: bool):
        """Context manager to turn augmentation on or off in a code block.

        Args:
            abool: Boolean value to set augmentation state.

        Example:
            ```python
            with dataset.augmentation(True):
                for i, data in enumerate(dataloader):
                    ...  # all data is augmented
            ```
        """
        augmentations = [
            "temporal_crop",
            "jitter",
            "rotate",
            "flip",
            "noise",
            "piecewise_resample",
            "linear_interpolate",
            "gamma_correct",
        ]
        states = {key: getattr(self, key).augment for key in augmentations}
        _augment = self.augment
        try:
            self.augment = abool
            yield
        finally:
            self.augment = _augment
            for key in augmentations:
                getattr(self, key).augment = states[key]

    @property
    def augment(self) -> bool:
        """Get the current augmentation state."""
        return self._augment

    @augment.setter
    def augment(self, value: bool) -> None:
        """Set the augmentation state and update augmentation callables.

        Args:
            value: Boolean value to set augmentation state.
        """
        self._augment = value
        if not self._augmentations_are_initialized:
            return
        # note: random_temporal_crop can override augment=True
        self.temporal_crop.random = self.random_temporal_crop if value else False
        self.jitter.augment = value
        self.rotate.augment = value
        self.flip.augment = value
        self.noise.augment = value
        # note: these two are not affected by augment
        self.piecewise_resample.augment = self.resampling
        self.linear_interpolate.augment = self.interpolate
        self.gamma_correct.augment = value

    def apply_augmentation(
            self,
            data: Dict[str, torch.Tensor],
            n_rot: Optional[int] = None,
            flip_axis: Optional[int] = None,
            contrast_factor: Optional[float] = None,
            brightness_factor: Optional[float] = None,
            gaussian_white_noise: Optional[float] = None,
            gamma: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Apply augmentation to a sample from the dataset.

        Args:
            data: Dictionary containing the sample data.
            n_rot: Number of rotations to apply.
            flip_axis: Axis to flip over.
            contrast_factor: Contrast factor for jitter augmentation.
            brightness_factor: Brightness factor for jitter augmentation.
            gaussian_white_noise: Standard deviation for noise augmentation.
            gamma: Gamma value for gamma correction.

        Returns:
            Dictionary containing the augmented sample data.
        """

        self.set_augmentation_params(
            n_rot=n_rot,
            flip_axis=flip_axis,
            contrast_factor=contrast_factor,
            brightness_factor=brightness_factor,
            gaussian_white_noise=gaussian_white_noise,
            gamma=gamma,
            start_frame=None,
            total_sequence_length=data["lum"].shape[0],
        )

        def transform_lum(lum):
            return self.piecewise_resample(
                self.rotate(
                    self.flip(
                        self.jitter(
                            self.noise(self.temporal_crop(lum)),
                        ),
                    )
                )
            )

        return {"lum": transform_lum(data["lum"])}

    def original_sequence_index(self, key: int) -> int:
        """Get the original sequence index from an index of the split.

        Args:
            key: Index of the split.

        Returns:
            Original sequence index.

        Raises:
            ValueError: If the key is not found in splits.
        """
        for index, splits in self.meta.sequence_index_to_splits.items():
            if key in splits:
                return index
        raise ValueError(f"key {key} not found in splits")

    def original_train_and_validation_indices(self) -> Tuple[List[int], List[int]]:
        """Get original training and validation indices for the dataloader.

        Returns:
            Tuple containing lists of train and validation indices.
        """
        return original_train_and_validation_indices(self)


class AugmentedDavis(MultiTaskDavis):
    """DAVIS dataset with controlled, rich augmentation.
    ... (docstring unchanged, omitted for brevity)
    """

    cached_sequences: List[Dict[str, torch.Tensor]]
    valid_flip_axes: List[int] = [0, 1, 2, 3]
    valid_rotations: List[int] = [0, 1, 2, 3, 4, 5]  # Indices for 60° increments

    def __init__(
            self,
            root_dir: Union[str, Path],
            n_frames: int = 19,
            flip_axes: List[int] = [0, 1],
            n_rotations: List[int] = [0, 1, 2, 3, 4, 5],
            build_stim_on_init: bool = True,
            temporal_split: bool = False,
            augment: bool = True,
            dt: float = 1 / 50,
            tasks: List[Literal["lum"]] = ["lum"],
            interpolate: bool = True,
            all_frames: bool = False,
            random_temporal_crop: bool = False,
            boxfilter: Dict[str, int] = dict(extent=15, kernel_size=13),
            vertical_splits: int = 3,
            contrast_std: Optional[float] = None,
            brightness_std: Optional[float] = None,
            gaussian_white_noise: Optional[float] = None,
            gamma_std: Optional[float] = None,
            center_crop_fraction: float = 0.7,
            indices: Optional[List[int]] = None,
            unittest: bool = False,
            shuffle_sequences: bool = True,
            **kwargs,
    ):
        if any([arg not in self.valid_flip_axes for arg in flip_axes]):
            raise ValueError(f"invalid flip axes {flip_axes}")

        # Handle both rotation indices (0-5) and degrees
        degree_to_index_map = {
            0: 0, 60: 1, 90: 1, 120: 2, 180: 3, 240: 4, 270: 4, 300: 5,
        }
        converted_rotations = []
        for rot in n_rotations:
            if rot in self.valid_rotations:
                converted_rotations.append(rot)
            elif rot in degree_to_index_map:
                converted_rotations.append(degree_to_index_map[rot])
            else:
                raise ValueError(
                    f"invalid rotation {rot}. Use indices 0-5 or degrees {list(degree_to_index_map.keys())}")
        n_rotations = converted_rotations

        super().__init__(
            root_dir=root_dir,
            tasks=tasks,
            interpolate=interpolate,
            n_frames=n_frames,
            dt=dt,
            augment=augment,
            all_frames=all_frames,
            resampling=True,
            random_temporal_crop=random_temporal_crop,
            boxfilter=boxfilter,
            vertical_splits=vertical_splits,
            p_flip=0,
            p_rot=0,
            contrast_std=contrast_std,
            brightness_std=brightness_std,
            gaussian_white_noise=gaussian_white_noise,
            gamma_std=gamma_std,
            center_crop_fraction=center_crop_fraction,
            unittest=unittest,
            _init_cache=True,
        )

        self.indices = np.array(indices) if indices is not None else None
        self.flip_axes = flip_axes
        self.n_rotations = n_rotations
        self.temporal_split = temporal_split
        self.shuffle_sequences = shuffle_sequences

        self.config.update({
            'root_dir': str(root_dir),
            'flip_axes': self.flip_axes,
            'n_rotations': self.n_rotations,
            'temporal_split': self.temporal_split,
            'shuffle_sequences': self.shuffle_sequences,
            'indices': self.indices,
            'center_crop_fraction': center_crop_fraction,
        })

        self._built = False
        if build_stim_on_init:
            self._build()
            self._built = True

    def _build(self):
        """Build augmented dataset with temporal splits and geometric augmentations."""
        # to deterministically apply temporal augmentation/binning of sequences
        # into ceil(sequence_length / n_frames) bins
        (
            self.cached_sequences,
            self.original_repeats,
        ) = temporal_split_cached_samples(
            self.cached_sequences, self.n_frames, split=self.temporal_split
        )

        vsplit_index, original_index, name = (
            self.arg_df[["index", "original_index", "name"]]
            .values.repeat(self.original_repeats, axis=0)
            .T
        )
        tsplit_index = np.arange(len(self.cached_sequences))

        n_frames = [d["lum"].shape[0] for d in self.cached_sequences]

        self.params = [
            (*p[0], p[1], p[2])
            for p in list(
                product(
                    zip(
                        name,
                        original_index,
                        vsplit_index,
                        tsplit_index,
                        n_frames,
                    ),
                    self.flip_axes,
                    self.n_rotations,
                )
            )
        ]

        self.arg_df = pd.DataFrame(
            self.params,
            columns=[
                "name",
                "original_index",
                "vertical_split_index",
                "temporal_split_index",
                "frames",
                "flip_ax",
                "n_rot",
            ],
        )

        # apply deterministic geometric augmentation
        cached_sequences = {}
        for i, (_, _, _, sample, _, flip_ax, n_rot) in enumerate(self.params):
            self.flip.axis = flip_ax
            self.rotate.n_rot = n_rot
            cached_sequences[i] = {
                key: self.rotate(self.flip(value))
                for key, value in self.cached_sequences[sample].items()
            }

        # Convert to list for easier shuffling
        self.cached_sequences = [cached_sequences[i] for i in sorted(cached_sequences.keys())]

        # SHUFFLE the sequences to mix different rotations/flips of different base sequences
        if self.shuffle_sequences:
            import random
            random.seed(42)  # Reproducible shuffling
            random.shuffle(self.cached_sequences)
            logger.info(f"Shuffled {len(self.cached_sequences)} augmented sequences for better variety")

        if self.indices is not None:
            # Apply indices selection after shuffling
            if len(self.indices) <= len(self.cached_sequences):
                self.cached_sequences = [self.cached_sequences[i] for i in self.indices if
                                         i < len(self.cached_sequences)]
                self.arg_df = self.arg_df.iloc[self.indices] if hasattr(self, 'arg_df') else None
                self.params = [self.params[i] for i in self.indices if i < len(self.params)]
            else:
                logger.warning(
                    f"Requested indices {self.indices} exceed available sequences {len(self.cached_sequences)}")

        # disable deterministically applied augmentation, such that in case
        # self.augment is True, the other augmentation types can be applied
        # randomly
        self.flip.augment = False
        self.rotate.augment = False
        # default to cropping 0 to n_frames
        self.temporal_crop.random = False
        if self.temporal_split:
            self.temporal_crop.augment = False

    def _original_length(self) -> int:
        """Return the original number of sequences before splitting."""
        return len(self) // self.vertical_splits

    def pad_nans(
            self, data: Dict[str, torch.Tensor], pad_to_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Pad the data with NaNs to a specified length.

        Args:
            data: Dictionary containing the data to pad.
            pad_to_length: Length to pad the data to.

        Returns:
            Padded data dictionary.
        """
        if pad_to_length is not None:
            data = {}
            for key, value in data.items():
                # pylint: disable=not-callable
                data[key] = nnf.pad(
                    value,
                    pad=(0, 0, 0, 0, 0, pad_to_length),
                    mode="constant",
                    value=np.nan,
                )
            return data
        return data

    def get_item(
            self, key: int, pad_to_length: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """Get a single item from the dataset.

        Args:
            key: Index of the item to retrieve.
            pad_to_length: Length to pad the data to.

        Returns:
            Dictionary containing the retrieved data.
        """
        if self.augment:
            return self.pad_nans(
                self.apply_augmentation(self.cached_sequences[key], n_rot=0, flip_axis=0),
                pad_to_length,
            )
        return self.pad_nans(self.cached_sequences[key], pad_to_length)