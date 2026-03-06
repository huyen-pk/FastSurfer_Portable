from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from inference_service import _extract_progress_values_from_line


def test_extract_progress_values_from_tqdm_line_with_percent_and_fraction() -> None:
    line = " 71%|████████████████████████████████████               | 181/256 [02:20<01:01,  1.22batch/s]"
    progress_values = _extract_progress_values_from_line(line)

    assert progress_values
    assert max(progress_values) == 71


def test_extract_progress_values_from_concatenated_tqdm_updates() -> None:
    line = (
        "  2%|█▏                                                   | 6/256 [00:04<03:20,  1.25batch/s]"
        "  2%|█▏                                                   | 6/256 [00:05<03:57,  1.05batch/s]"
    )
    progress_values = _extract_progress_values_from_line(line)

    assert progress_values == [2]


def test_extract_progress_values_from_non_progress_log() -> None:
    line = "[INFO: dataset.py:   71]: Loading Sagittal with input voxelsize [1. 1.]"

    assert _extract_progress_values_from_line(line) == []
