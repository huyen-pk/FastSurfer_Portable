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

"""Unit tests for DI inference engine factory."""

import os
import unittest
from unittest.mock import MagicMock, patch

from di.factories import InferenceEngineFactory


class TestInferenceEngineFactory(unittest.TestCase):
    """Test cases for InferenceEngineFactory."""

    @patch("di.factories.Inference")
    def test_create_inference_engine_respects_disable_onnx(self, mock_inference):
        """Test disabling ONNX forces PyTorch inference engine."""
        cfg = MagicMock()
        factory = InferenceEngineFactory()

        with patch.dict(os.environ, {"FASTSURFER_DISABLE_ONNX": "1"}, clear=False):
            engine = factory.create_inference_engine(cfg)

        self.assertEqual(engine, mock_inference.return_value)
        mock_inference.assert_called_once()


if __name__ == "__main__":
    unittest.main()
