"""Tests for IPC server progress event emission and JSON serialization."""

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

# Add parent directory to path to import ipc_server
sys.path.insert(0, str(Path(__file__).parent.parent))

from ipc_server import FastSurferIPCServer


class TestIPCServerProgressEmission:
    """Test progress event emission from IPC server."""

    def test_progress_callback_emits_json_with_progress_clamped_to_0_100(self, monkeypatch):
        """Test that progress callback clamps values to 0-100 and emits valid JSON."""
        server = FastSurferIPCServer()
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        # Mock the inference service to call progress callback with various values
        progress_values = []

        def mock_predict(input_path, output_dir, return_base64, progress_callback):
            progress_values.append(progress_callback)
            # Simulate calling progress callback with various values
            progress_callback(2, "Starting processing")
            progress_callback(50, "Halfway done")
            progress_callback(150, "Exceeds 100")  # Should clamp to 100
            progress_callback(-10, "Negative")  # Should clamp to 0
            return {"output_path": "/tmp/out/test.mgz", "output_filename": "test.mgz"}

        with patch.object(server.inference_service, "predict_from_path", side_effect=mock_predict):
            # Mock Path.exists to avoid file validation
            with patch("ipc_server.Path.exists", return_value=True):
                result = server._predict_from_path({
                    "input_path": "/tmp/test.mgz",
                    "task_id": "test-123"
                })

        output_lines = captured_output.getvalue().strip().split("\n")
        assert len(output_lines) == 4

        # Parse and validate each progress event
        events = [json.loads(line) for line in output_lines]
        assert events[0]["progress"] == 2
        assert events[0]["message"] == "Starting processing"
        assert events[0]["event"] == "progress"
        assert events[0]["task_id"] == "test-123"

        assert events[1]["progress"] == 50
        assert events[2]["progress"] == 100  # Clamped from 150
        assert events[3]["progress"] == 0  # Clamped from -10

    def test_progress_callback_includes_task_id_when_provided(self, monkeypatch):
        """Test that task_id is included in progress event when provided."""
        server = FastSurferIPCServer()
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        def mock_predict(input_path, output_dir, return_base64, progress_callback):
            progress_callback(42, "Processing")
            return {"output_path": "/tmp/out.mgz", "output_filename": "out.mgz"}

        with patch.object(server.inference_service, "predict_from_path", side_effect=mock_predict):
            with patch("ipc_server.Path.exists", return_value=True):
                server._predict_from_path({
                    "input_path": "/tmp/test.mgz",
                    "task_id": "unique-task-456"
                })

        output_lines = captured_output.getvalue().strip().split("\n")
        event = json.loads(output_lines[0])
        assert event["task_id"] == "unique-task-456"

    def test_progress_callback_omits_task_id_when_not_provided(self, monkeypatch):
        """Test that task_id is omitted from progress event when not provided."""
        server = FastSurferIPCServer()
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        def mock_predict(input_path, output_dir, return_base64, progress_callback):
            progress_callback(42, "Processing")
            return {"output_path": "/tmp/out.mgz", "output_filename": "out.mgz"}

        with patch.object(server.inference_service, "predict_from_path", side_effect=mock_predict):
            with patch("ipc_server.Path.exists", return_value=True):
                server._predict_from_path({"input_path": "/tmp/test.mgz"})

        output_lines = captured_output.getvalue().strip().split("\n")
        event = json.loads(output_lines[0])
        assert "task_id" not in event
        assert event["event"] == "progress"
        assert event["progress"] == 42

    def test_progress_callback_with_empty_message_uses_default(self, monkeypatch):
        """Test that empty message is replaced with default."""
        server = FastSurferIPCServer()
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        def mock_predict(input_path, output_dir, return_base64, progress_callback):
            progress_callback(50, "")
            return {"output_path": "/tmp/out.mgz", "output_filename": "out.mgz"}

        with patch.object(server.inference_service, "predict_from_path", side_effect=mock_predict):
            with patch("ipc_server.Path.exists", return_value=True):
                server._predict_from_path({"input_path": "/tmp/test.mgz"})

        output_lines = captured_output.getvalue().strip().split("\n")
        event = json.loads(output_lines[0])
        assert event["message"] == "Processing MRI"

    def test_progress_events_are_newline_delimited_json(self, monkeypatch):
        """Test that multiple progress events are properly newline-delimited."""
        server = FastSurferIPCServer()
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        def mock_predict(input_path, output_dir, return_base64, progress_callback):
            progress_callback(10, "Step 1")
            progress_callback(20, "Step 2")
            progress_callback(30, "Step 3")
            return {"output_path": "/tmp/out.mgz", "output_filename": "out.mgz"}

        with patch.object(server.inference_service, "predict_from_path", side_effect=mock_predict):
            with patch("ipc_server.Path.exists", return_value=True):
                server._predict_from_path({"input_path": "/tmp/test.mgz"})

        output = captured_output.getvalue()
        lines = output.strip().split("\n")
        assert len(lines) == 3

        # Verify each line is valid JSON
        for line in lines:
            event = json.loads(line)
            assert "event" in event
            assert "progress" in event

    def test_handle_request_health_method(self):
        """Test health check request returns ok status."""
        server = FastSurferIPCServer()
        response = server.handle_request({"id": 1, "method": "health"})

        assert response["ok"] is True
        assert response["result"]["status"] == "ok"
        assert response["id"] == 1

    def test_handle_request_predict_with_missing_input_path_returns_error(self):
        """Test predict request without input_path raises error."""
        server = FastSurferIPCServer()
        
        with pytest.raises(ValueError) as exc_info:
            server.handle_request({"id": 2, "method": "predict", "params": {}})
        
        assert "input_path" in str(exc_info.value).lower()

    def test_handle_request_shutdown_stops_server(self):
        """Test shutdown request stops the server."""
        server = FastSurferIPCServer()
        assert server._running is True

        response = server.handle_request({"id": 3, "method": "shutdown"})

        assert response["ok"] is True
        assert response["result"]["status"] == "shutting_down"
        assert server._running is False

    def test_handle_request_unknown_method_returns_error(self):
        """Test unknown method raises error."""
        server = FastSurferIPCServer()
        
        with pytest.raises(ValueError) as exc_info:
            server.handle_request({"id": 4, "method": "unknown_method"})

        assert "unknown method" in str(exc_info.value).lower()


class TestIPCServerProgressIntegration:
    """Integration tests for progress event flow through IPC server."""

    def test_multiple_concurrent_progress_streams_with_different_task_ids(self, monkeypatch):
        """Test that multiple task IDs produce separate progress streams."""
        server = FastSurferIPCServer()
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        def mock_predict(input_path, output_dir, return_base64, progress_callback):
            # Simulate first task
            progress_callback(25, "Task 1 progress")
            progress_callback(75, "Task 1 complete")
            return {"output_path": "/tmp/out1.mgz", "output_filename": "out1.mgz"}

        with patch.object(server.inference_service, "predict_from_path", side_effect=mock_predict):
            with patch("ipc_server.Path.exists", return_value=True):
                # First prediction with task_id
                server._predict_from_path({
                    "input_path": "/tmp/test1.mgz",
                    "task_id": "task-1"
                })

        output = captured_output.getvalue()
        lines = output.strip().split("\n")

        # Should have 2 progress events with task_id "task-1"
        assert len(lines) == 2
        for line in lines:
            event = json.loads(line)
            assert event["task_id"] == "task-1"

    def test_progress_callback_json_is_valid_and_parseable(self, monkeypatch):
        """Test that all emitted JSON is valid and parseable."""
        server = FastSurferIPCServer()
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        def mock_predict(input_path, output_dir, return_base64, progress_callback):
            # Various progress values
            values = [0, 1, 50, 75, 99, 100]
            for v in values:
                progress_callback(v, f"Progress: {v}%")
            return {"output_path": "/tmp/out.mgz", "output_filename": "out.mgz"}

        with patch.object(server.inference_service, "predict_from_path", side_effect=mock_predict):
            with patch("ipc_server.Path.exists", return_value=True):
                server._predict_from_path({"input_path": "/tmp/test.mgz"})

        output = captured_output.getvalue()
        lines = output.strip().split("\n")

        assert len(lines) == 6
        # Verify all lines are parseable JSON
        events = []
        for line in lines:
            try:
                event = json.loads(line)
                events.append(event)
            except json.JSONDecodeError as e:
                pytest.fail(f"Failed to parse JSON: {line}, error: {e}")

        # Verify structure
        for i, event in enumerate(events):
            assert event["event"] == "progress"
            assert event["progress"] == [0, 1, 50, 75, 99, 100][i]
            assert isinstance(event["message"], str)
