# BDD Requirements: Progress Reporting for Native Inference

## Feature: Real-time Progress Tracking for Rust ONNX Inference
As a user performing brain segmentations
I want to see the progress bar update in real-time
So that I know the application is still working and how much work is left.

### Acceptance Criteria

#### 1. Initialization Accuracy
**Scenario: Starting a native inference task**
* **Given** a valid MRI volume input
* **And** the Rust ONNX native engine is selected
* **When** I start the inference task
* **Then** the system should immediately emit a `started` event
* **And** the event must contain the correct `total` slice count derived from the input volume.

#### 2. Real-time Async Updates
**Scenario: Processing slices sequentially within planes**
* **Given** a system processing slices sequentially (to match legacy `SequentialSampler` behavior)
* **When** each slice completes its inference pass
* **Then** `item_progress` events should be emitted asynchronously without waiting for the entire plane to finish
* **And** these events should arrive *during* the execution interval of the inference task.

#### 3. Progress Monotonicity
**Scenario: Sequential progress updates**
* **Given** multiple progress events being emitted from parallel threads
* **When** they are received by the UI
* **Then** the reported [progress](file:///home/huyenpk/Projects/FastSurfer/app/gui/workbench/src-tauri/src/prediction.rs#11-15) percentage must be monotonically increasing
* **And** the `completed` slice count must never decrease.

#### 4. Completion Finalization
**Scenario: Successfully finishing a task**
* **Given** all slices for all planes (Coronal, Axial, Sagittal) have been processed
* **When** the logits fusion is completed
* **Then** a `completed` event must be emitted
* **And** the [progress](file:///home/huyenpk/Projects/FastSurfer/app/gui/workbench/src-tauri/src/prediction.rs#11-15) must be exactly [100](file:///home/huyenpk/Projects/FastSurfer/app/gui/workbench/src-tauri/testing/rust/controller_tests/progress_tests.rs#30-51).

## Real Dependency Requirement
- All tests for these scenarios **MUST** use the real `Subject140` MRI data.
- **NO MOCKS** or fake progress events are allowed in these behavioral tests.
