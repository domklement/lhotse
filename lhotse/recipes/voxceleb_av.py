"""
The Oxford-BBC Lip Reading Sentences 2 (voxceleb_av) Dataset
The dataset consists of thousands of spoken sentences from BBC television.
For more details see: https://www.robots.ox.ac.uk/~vgg/data/lip_reading/voxceleb_av.html
"""

import logging
import os
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import tqdm
from torchcodec.decoders import AudioDecoder, VideoDecoder
from tqdm import tqdm

from lhotse import AudioSource, Recording, SupervisionSegment, load_manifest
from lhotse.audio.utils import VideoInfo
from lhotse.cut import CutSet, MonoCut
from lhotse.utils import Pathlike, safe_extract


def extract_single_tar(tar_file: Path, output_base: Path):
    """
    Extract a single tar file and fix permissions.

    Args:
        tar_file: Path to tar file
        output_base: Base output directory

    Returns:
        Tuple of (extract_dir, success, error_message)
    """
    # Extract dataset part name
    filename = tar_file.stem  # Remove .tar

    # Remove the trailing number part (e.g., -001, -002)
    parts = filename.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        dset_part = parts[0]
    else:
        dset_part = filename

    # Create output directory
    extract_dir = output_base / dset_part
    extract_dir.mkdir(parents=True, exist_ok=True)

    # Extract tar file
    with tarfile.open(tar_file, "r") as tar:
        safe_extract(tar, extract_dir)


def untar_voxceleb_av_files(
    tar_pattern: str,
    output_dir: Optional[Pathlike] = None,
    num_workers: Optional[int] = None,
):
    """
    Untar all voxceleb_av tar files matching the pattern in parallel.

    Each tar file is extracted to a directory based on its name.
    For example: voxceleb_av-train-001.tar -> voxceleb_av-train/

    Args:
        tar_pattern: Glob pattern for tar files
        output_dir: Optional output directory (defaults to parent of tar files)
        num_workers: Number of parallel workers (defaults to CPU count)
    """
    # Parse the pattern to get directory and glob
    completed_detector = output_dir / ".completed"
    if completed_detector.is_file():
        logging.info("voxceleb_av has been extracted already, skipping.")
        return

    pattern_path = Path(tar_pattern)
    tar_dir = pattern_path.parent
    glob_pattern = pattern_path.name

    if not tar_dir.exists():
        raise ValueError(f"Directory does not exist: {tar_dir}")

    # Find all matching tar files
    tar_files = sorted(tar_dir.glob(glob_pattern))

    if not tar_files:
        raise ValueError(f"No tar files found matching: {tar_pattern}")

    logging.info(f"Found {len(tar_files)} tar files to extract")

    # Determine output directory
    if output_dir is None:
        output_base = tar_dir
    else:
        output_base = Path(output_dir)
        output_base.mkdir(parents=True, exist_ok=True)

    # Parallel extraction
    extract_func = partial(extract_single_tar, output_base=output_base)

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(extract_func, tar_file): tar_file for tar_file in tar_files
        }

        # Process results with progress bar
        with tqdm.tqdm(total=len(tar_files), desc="Extracting") as pbar:
            for future in as_completed(futures):
                future.result()
                pbar.update(1)

    completed_detector.touch()
    logging.info(f"\nExtraction complete!")


def download_voxceleb_av(
    target_dir: Pathlike = ".", force_download: Optional[bool] = False
) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as import_error:
        raise RuntimeError(
            "huggingface_hub is required for voxceleb_av downloads. Install it via:\n"
            "  pip install huggingface_hub\n"
        ) from import_error

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    download_dir = target_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    download_patterns = [
        f"vox2/*.tar",
    ]
    snapshot_download(
        repo_id="nguyenvulebinh/AVYT",
        repo_type="dataset",
        local_dir=download_dir,
        force_download=bool(force_download),
        allow_patterns=download_patterns,
    )

    untar_voxceleb_av_files(f"{download_dir}/{download_patterns[0]}", target_dir)

    return target_dir


def find_video_files(dataset_dir: Path) -> List[Path]:
    """
    Find all .video files in the given directory.

    Args:
        dataset_dir: Path to the dataset directory

    Returns:
        List of paths to .video files
    """
    video_files = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(".video"):
                video_files.append(Path(root) / file)
    return video_files


def read_label_file(label_path: Path) -> str:
    """
    Read the transcript from a .label file.

    Args:
        label_path: Path to the .label file

    Returns:
        Transcript text
    """
    with open(label_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def read_sample_id_file(sample_id_path: Path) -> str:
    """
    Read the sample ID from a .sample_id file.

    Args:
        sample_id_path: Path to the .sample_id file

    Returns:
        Sample ID string (format: "main/speaker_id/session_id")
    """
    with open(sample_id_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def create_cut_from_video(video_path: Path, dataset_part: str) -> Optional[MonoCut]:
    """
    Create a Lhotse MonoCut from a video file and its associated metadata.

    Args:
        video_path: Path to the .video file
        dataset_part: Name of the dataset partition

    Returns:
        MonoCut object or None if processing fails
    """
    # Construct paths to associated files
    label_path = video_path.with_suffix(".label")
    sample_id_path = video_path.with_suffix(".sample_id")
    base_name = video_path.stem

    transcript = read_label_file(label_path)
    sample_id = read_sample_id_file(sample_id_path)

    speaker_id = sample_id.split("/")[1]

    # Create a unique ID for the cut
    cut_id = (
        f"{dataset_part}_{sample_id.replace('/', '_')}"
        if sample_id
        else f"{dataset_part}_{base_name}"
    )

    adec = AudioDecoder(video_path)
    vdec = VideoDecoder(video_path)

    audio_samples = adec.get_all_samples()

    recording = Recording(
        id=cut_id,
        sources=[
            AudioSource(
                type="file",
                channels=[0],
                source=str(video_path),
                video=VideoInfo(
                    fps=int(round(vdec.metadata.average_fps)),
                    num_frames=vdec.metadata.num_frames,
                    height=vdec.metadata.height,
                    width=vdec.metadata.width,
                ),
            )
        ],
        sampling_rate=adec.metadata.sample_rate,
        num_samples=audio_samples.data.shape[-1],
        duration=audio_samples.duration_seconds,
    )

    # Create Supervision
    supervision = SupervisionSegment(
        id=cut_id,
        recording_id=cut_id,
        start=0.0,
        duration=audio_samples.duration_seconds,  # Will be populated when audio is loaded
        channel=0,
        text=transcript,
        language="en",
        speaker=speaker_id,
        custom={"video_path": str(video_path)},
    )

    # Create MonoCut
    cut = MonoCut(
        id=cut_id,
        start=0.0,
        duration=audio_samples.duration_seconds,  # Will be populated when audio is loaded
        channel=0,
        supervisions=[supervision],
        recording=recording,
        custom={"dataset_part": dataset_part, "sample_id": sample_id},
    )

    return cut


def process_dataset_part(dataset_dir: Path, part_name: str, num_workers) -> CutSet:
    """
    Process a single dataset partition and create a CutSet.

    Args:
        dataset_dir: Path to the dataset partition directory
        part_name: Name of the partition (e.g., 'train', 'val', 'test')
        num_workers: Number of parallel workers to use (default: 1 for sequential processing)

    Returns:
        CutSet containing all cuts for this partition
    """
    logging.info(f"\nProcessing dataset part: {part_name}")

    # Find all video files
    video_files = find_video_files(dataset_dir)
    logging.info(f"Found {len(video_files)} video files")

    process_func = partial(create_cut_from_video, dataset_part=part_name)

    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better memory efficiency with large datasets
        cuts = list(
            tqdm(
                pool.imap_unordered(process_func, video_files),
                total=len(video_files),
                desc=f"Creating cuts for {part_name}",
            )
        )

    return CutSet.from_cuts(cuts)


def prepare_voxceleb_av(
    corpus_dir: Pathlike,
    dataset_parts: Optional[Sequence[str]] = None,
    output_dir: Optional[Pathlike] = None,
    num_jobs: int = 1,
) -> Dict[str, CutSet]:
    if dataset_parts is None:
        dataset_parts = ["dev", "test"]

    os.makedirs(output_dir, exist_ok=True)
    cutsets = {}
    # Process each dataset partition
    for part in dataset_parts:

        # Check if manifest already exists and skip if requested
        output_filename = f"{part}_cuts.jsonl.gz"
        output_path = Path(output_dir) / output_filename

        if output_path.exists():
            logging.info(f"Loading {part}: Manifest already exists at {output_path}")
            cutsets[part] = load_manifest(output_path)
            continue

        # Process the partition with parallel processing
        cutset = process_dataset_part(
            Path(corpus_dir) / part, part, num_workers=num_jobs
        )

        # Save the CutSet
        logging.info(f"Saving CutSet to: {output_path}")
        cutset.to_file(output_path)
        cutsets[part] = cutset
    return cutsets
