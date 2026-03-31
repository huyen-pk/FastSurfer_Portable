// This test suite integrates with test-containers for environment isolation.
#[path = "controller_tests/support.rs"]
mod support;

#[path = "controller_tests/transport_tests.rs"]
mod transport_tests;

#[path = "controller_tests/data_loading_tests.rs"]
mod data_loading_tests;

#[path = "controller_tests/file_management_tests.rs"]
mod file_management_tests;

#[path = "controller_tests/backend_resolution_tests.rs"]
mod backend_resolution_tests;

#[path = "controller_tests/input_validation_tests.rs"]
mod input_validation_tests;

#[path = "controller_tests/pre_inference_processing_tests.rs"]
mod pre_inference_processing_tests;

#[path = "controller_tests/post_inference_processing_tests.rs"]
mod post_inference_processing_tests;

#[path = "controller_tests/native_progress_behavior_test.rs"]
mod native_progress_behavior_test;
