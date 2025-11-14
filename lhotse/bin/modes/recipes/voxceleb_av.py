from typing import Sequence

import click

from lhotse.bin.modes import download, prepare
from lhotse.recipes.voxceleb_av import download_voxceleb_av, prepare_voxceleb_av
from lhotse.utils import Pathlike

__all__ = ["voxceleb_av"]


@prepare.command(context_settings=dict(show_default=True))
@click.argument("corpus_dir", type=click.Path(exists=True, dir_okay=True))
@click.argument("output_dir", type=click.Path())
@click.option(
    "-p",
    "--dataset-parts",
    type=str,
    default=("dev", "test"),
    multiple=True,
    help="List of dataset parts to prepare. To prepare multiple parts, pass each with `-p` "
    "Example: `-p dev -p test`",
)
@click.option(
    "-j",
    "--num-jobs",
    type=int,
    default=1,
    help="How many threads to use (can give good speed-ups with slow disks).",
)
def voxceleb_av(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
    dataset_parts: Sequence[str],
    num_jobs: int,
):
    prepare_voxceleb_av(
        corpus_dir,
        output_dir=output_dir,
        dataset_parts=dataset_parts,
        num_jobs=num_jobs,
    )


@download.command(context_settings=dict(show_default=True))
@click.argument("target_dir", type=click.Path())
@click.option(
    "--force-download",
    type=bool,
    default=False,
    help="If True, download even if file is present.",
)
def voxceleb_av(target_dir: Pathlike, force_download: bool):
    """(Mini) Librispeech download."""
    download_voxceleb_av(target_dir, force_download)
