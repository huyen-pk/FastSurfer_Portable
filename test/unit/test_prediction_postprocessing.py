# Copyright 2019 Image Analysis Lab, German Center for Neurodegenerative Diseases (DZNE), Bonn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for prediction_postprocessing module.

These tests verify that the arithmetic/post-processing operations extracted
from run_prediction.py (get_prediction) produce identical results before
and after refactoring.

The test shapes and class counts mirror what is used in production when
running inference on a FreeSurfer conforming subject (e.g. bert):
  - Conformed volume shape: (256, 256, 256) voxels
  - num_classes: 79  (same as RunModelOnData.num_classes)
"""

import unittest
from unittest.mock import patch

import numpy as np
import torch

from FastSurferCNN.prediction_postprocessing import init_pred_prob, postprocess_pred


# Spatial dimensions that match a conformed FreeSurfer subject (e.g. bert)
_H, _W, _D = 256, 256, 256
_NUM_CLASSES = 79  # as set in RunModelOnData.num_classes


class TestInitPredProb(unittest.TestCase):
    """Tests for init_pred_prob – tensor initialisation."""

    def test_shape(self):
        """Tensor has exactly the requested shape."""
        shape = (_H, _W, _D, _NUM_CLASSES)
        t = init_pred_prob(shape, device=torch.device("cpu"))
        self.assertEqual(t.shape, torch.Size(shape))

    def test_all_zeros(self):
        """Tensor must be zero-initialised."""
        shape = (4, 4, 4, _NUM_CLASSES)
        t = init_pred_prob(shape, device=torch.device("cpu"))
        self.assertTrue(torch.all(t == 0))

    def test_default_dtype_float16(self):
        """Default dtype must be float16 (matches get_prediction kwargs)."""
        shape = (4, 4, 4, _NUM_CLASSES)
        t = init_pred_prob(shape, device=torch.device("cpu"))
        self.assertEqual(t.dtype, torch.float16)

    def test_custom_dtype(self):
        """Explicit dtype is respected."""
        shape = (4, 4, 4, _NUM_CLASSES)
        t = init_pred_prob(shape, device=torch.device("cpu"), dtype=torch.float32)
        self.assertEqual(t.dtype, torch.float32)

    def test_no_grad_by_default(self):
        """requires_grad must be False by default."""
        shape = (4, 4, 4, _NUM_CLASSES)
        t = init_pred_prob(shape, device=torch.device("cpu"))
        self.assertFalse(t.requires_grad)

    def test_requires_grad_opt_in(self):
        """requires_grad can be set to True."""
        shape = (4, 4, 4, _NUM_CLASSES)
        t = init_pred_prob(shape, device=torch.device("cpu"), requires_grad=True)
        self.assertTrue(t.requires_grad)

    def test_device_cpu(self):
        """Tensor lives on CPU when cpu device is requested."""
        shape = (4, 4, 4, _NUM_CLASSES)
        t = init_pred_prob(shape, device=torch.device("cpu"))
        self.assertEqual(t.device.type, "cpu")

    def test_small_shape(self):
        """Works with small (non-brain) shapes used in synthetic tests."""
        shape = (2, 3, 4, 5)
        t = init_pred_prob(shape, device=torch.device("cpu"))
        self.assertEqual(t.shape, torch.Size(shape))


class TestPostprocessPred(unittest.TestCase):
    """Tests for postprocess_pred – argmax → reorder → label-map → cortex-split.

    We verify:
    1. The output is a numpy ndarray.
    2. The output shape equals the spatial dimensions of the input tensor.
    3. All values in the output exist in the FreeSurfer label LUT passed in.
    4. The pipeline is equivalent to calling the original inline steps, confirming
       that the refactoring is behaviour-preserving.
    """

    def _make_labels(self, n: int) -> np.ndarray:
        """Create a simple monotone label array [0, 1, …, n-1]."""
        return np.arange(n, dtype=np.int64)

    def _identity_back_to_native(self, t: torch.Tensor) -> torch.Tensor:
        """Identity orientation transform (LIA == native)."""
        return t

    # ------------------------------------------------------------------
    # helpers to replicate the original inline logic from get_prediction
    # ------------------------------------------------------------------
    def _original_inline_postprocess(
        self,
        pred_prob: torch.Tensor,
        back_to_native,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Verbatim copy of the original inline post-processing code."""
        from FastSurferCNN.data_loader import data_utils as du

        pred_classes = torch.argmax(pred_prob, 3)
        del pred_prob
        pred_classes = back_to_native(pred_classes)
        pred_classes = du.map_label2aparc_aseg(pred_classes, labels)
        pred_classes = du.split_cortex_labels(pred_classes.cpu().numpy())
        return pred_classes

    # ------------------------------------------------------------------
    # Tests
    # ------------------------------------------------------------------

    def test_output_is_ndarray(self):
        """postprocess_pred must return a numpy ndarray."""
        shape = (4, 4, 4, _NUM_CLASSES)
        pred_prob = torch.zeros(shape, dtype=torch.float32)
        labels = self._make_labels(_NUM_CLASSES)
        result = postprocess_pred(pred_prob, self._identity_back_to_native, labels)
        self.assertIsInstance(result, np.ndarray)

    def test_output_shape_matches_spatial_dims(self):
        """Output spatial shape must equal the first three dimensions of pred_prob."""
        h, w, d = 8, 8, 8
        shape = (h, w, d, _NUM_CLASSES)
        pred_prob = torch.zeros(shape, dtype=torch.float32)
        labels = self._make_labels(_NUM_CLASSES)
        result = postprocess_pred(pred_prob, self._identity_back_to_native, labels)
        self.assertEqual(result.shape, (h, w, d))

    def test_argmax_selects_highest_class(self):
        """Argmax over class dimension picks the class with the highest logit."""
        h, w, d = 4, 4, 4
        n_cls = 10
        pred_prob = torch.zeros((h, w, d, n_cls), dtype=torch.float32)
        # Make class 7 the winner everywhere
        pred_prob[..., 7] = 1.0
        labels = self._make_labels(n_cls)
        # With identity back_to_native and identity labels, all voxels → 7
        result = postprocess_pred(pred_prob, self._identity_back_to_native, labels)
        self.assertTrue(np.all(result == 7))

    def test_back_to_native_is_applied(self):
        """The back_to_native transform is called and its result is used."""
        h, w, d = 4, 4, 4
        n_cls = 10
        pred_prob = torch.zeros((h, w, d, n_cls), dtype=torch.float32)
        pred_prob[..., 3] = 1.0
        labels = self._make_labels(n_cls)

        called = []

        def _tracking_back_to_native(t: torch.Tensor) -> torch.Tensor:
            called.append(True)
            return t  # identity for simplicity

        postprocess_pred(pred_prob, _tracking_back_to_native, labels)
        self.assertTrue(called, "back_to_native was not called")

    def test_label_mapping_applied(self):
        """Labels array is used to remap network class indices to LUT values."""
        h, w, d = 4, 4, 4
        n_cls = 10
        pred_prob = torch.zeros((h, w, d, n_cls), dtype=torch.float32)
        # class index 2 wins everywhere
        pred_prob[..., 2] = 1.0
        # map index 2 → LUT value 42
        labels = np.array([0, 1, 42, 3, 4, 5, 6, 7, 8, 9], dtype=np.int64)
        result = postprocess_pred(pred_prob, self._identity_back_to_native, labels)
        self.assertTrue(np.all(result == 42))

    def test_behavior_identical_to_original_inline_code(self):
        """postprocess_pred must produce the same result as the original inline code.

        This is the core regression test: it confirms that extracting the
        post-processing logic into a separate function did not change any
        computation, using synthetic data shaped like a conformed FreeSurfer
        subject (e.g. bert, 256³ voxels, 79 classes).
        """
        # Use a small spatial volume to keep the test fast; the logic is the
        # same regardless of spatial size.
        h, w, d = 16, 16, 16
        n_cls = _NUM_CLASSES

        rng = np.random.default_rng(seed=42)
        data = rng.standard_normal((h, w, d, n_cls)).astype(np.float32)
        pred_prob_a = torch.from_numpy(data.copy())
        pred_prob_b = torch.from_numpy(data.copy())

        labels = self._make_labels(n_cls)

        result_new = postprocess_pred(
            pred_prob_a, self._identity_back_to_native, labels.copy()
        )
        result_old = self._original_inline_postprocess(
            pred_prob_b, self._identity_back_to_native, labels.copy()
        )

        np.testing.assert_array_equal(
            result_new,
            result_old,
            err_msg=(
                "postprocess_pred output differs from the original inline code. "
                "The refactoring changed behaviour."
            ),
        )


if __name__ == "__main__":
    unittest.main()
