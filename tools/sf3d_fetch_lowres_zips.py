"""Fetch the ARKit lowres assets as ZIP ARCHIVES ONLY (no extraction).

Extraction is what makes this download slow, not bandwidth. Each video's
lowres_depth archive holds ~3-6k tiny PNGs; creating ~2.7M small files through
the MooseFS FUSE client ran at ~0.8 MB/s and did NOT speed up with more
workers (8 -> 24 workers gained nothing once the file count was halved by
dropping `confidence`), i.e. we sit on a metadata-op ceiling. Downloading the
archives themselves is network-bound at ~60 MB/s -- roughly two orders of
magnitude better.

The toolkit deletes each archive right after extracting it
(download_data.py:216 passes keep_zip=False), so this driver monkeypatches
`unzip_file` to a no-op. `download_assets_for_video_id` resolves that name
from its own module globals at call time, so patching the module attribute is
enough -- no fork of the toolkit required.

The archive then becomes the only artifact, and tools/sf3d_process.py reads
frames straight out of it (see LowresSource there), which also keeps the
volume at ~609 files instead of millions.

Videos whose lowres_depth directory already exists are skipped by the
toolkit's own resume check; LowresSource falls back to reading those from the
extracted directory, so a partially-extracted tree stays usable.

Run from the SceneFun3D toolkit root:

    python /workspace/reproc_test/sf3d_fetch_lowres_zips.py \
        --split train_val_set \
        --download_dir /workspace/scenefun3d/train_val \
        --dataset_assets lowres_depth lowres_wide_intrinsics lowres_poses \
        --workers 12
"""

import data_downloader.download_utils.download_data as dd


def _skip_unzip(file_name, dst, keep_zip=True):
    """Keep the .zip, create no files. Signature matches unzip_file."""
    return True


dd.unzip_file = _skip_unzip

import parallel_download  # noqa: E402  (must come after the patch is installed)


if __name__ == "__main__":
    print("[sf3d_fetch_lowres_zips] extraction disabled; archives will be kept")
    parallel_download.main()
