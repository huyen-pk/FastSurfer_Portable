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

"""Unit tests for DI device manager."""

import unittest
from unittest.mock import patch

import torch

from FastSurferCNN.di import DeviceManager


class TestDeviceManager(unittest.TestCase):
    """Test cases for DeviceManager."""

    @patch("torch.cuda.is_available")
    def test_device_manager_auto_cuda(self, mock_cuda_available):
        """Test automatic CUDA device selection when available."""
        mock_cuda_available.return_value = True
        dm = DeviceManager()
        device = dm.get_device()
        self.assertEqual(device.type, "cuda")

    @patch("torch.cuda.is_available")
    def test_device_manager_auto_cpu(self, mock_cuda_available):
        """Test automatic CPU device selection when CUDA unavailable."""
        mock_cuda_available.return_value = False
        dm = DeviceManager()
        device = dm.get_device()
        self.assertEqual(device.type, "cpu")

    def test_device_manager_explicit_device(self):
        """Test explicit device specification."""
        cpu_device = torch.device("cpu")
        dm = DeviceManager(device=cpu_device)
        device = dm.get_device()
        self.assertEqual(device.type, "cpu")

    @patch("torch.cuda.is_available")
    def test_is_cuda_available(self, mock_cuda_available):
        """Test CUDA availability check."""
        mock_cuda_available.return_value = True
        dm = DeviceManager()
        self.assertTrue(dm.is_cuda_available())
        
        mock_cuda_available.return_value = False
        self.assertFalse(dm.is_cuda_available())

    @patch("torch.cuda.device_count")
    def test_device_count(self, mock_device_count):
        """Test device count method."""
        mock_device_count.return_value = 2
        dm = DeviceManager()
        self.assertEqual(dm.device_count(), 2)

    @patch("torch.cuda.device_count")
    def test_is_parallel_capable_true(self, mock_device_count):
        """Test parallel capability check when multiple devices available."""
        mock_device_count.return_value = 2
        cuda_device = torch.device("cuda")
        dm = DeviceManager(device=cuda_device)
        self.assertTrue(dm.is_parallel_capable())

    @patch("torch.cuda.device_count")
    def test_is_parallel_capable_false_single_device(self, mock_device_count):
        """Test parallel capability check with single device."""
        mock_device_count.return_value = 1
        cuda_device = torch.device("cuda")
        dm = DeviceManager(device=cuda_device)
        self.assertFalse(dm.is_parallel_capable())

    @patch("torch.cuda.device_count")
    def test_is_parallel_capable_false_cpu(self, mock_device_count):
        """Test parallel capability check with CPU device."""
        mock_device_count.return_value = 2
        cpu_device = torch.device("cpu")
        dm = DeviceManager(device=cpu_device)
        self.assertFalse(dm.is_parallel_capable())


if __name__ == "__main__":
    unittest.main()
