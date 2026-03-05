use super::support::next_test_id;
use crate::file_mgmt::validate_result_path;
use std::fs;
use std::path::Path;

#[test]
fn validate_empty_path_should_return_path_is_empty_error() {
    let result = validate_result_path("   ");

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Path is empty");
}

#[test]
fn validate_missing_path_should_return_path_does_not_exist_error() {
    let missing_path = std::env::temp_dir().join(format!(
        "desktop_test_missing_{}.nii.gz",
        next_test_id()
    ));
    let result = validate_result_path(&missing_path.to_string_lossy());

    assert!(result.is_err());
    assert!(result.unwrap_err().starts_with("Path does not exist:"));
}

#[test]
fn validate_existing_file_path_should_return_canonical_file_path() {
    let temp_file = std::env::temp_dir().join(format!(
        "desktop_test_existing_file_{}.txt",
        next_test_id()
    ));
    fs::write(&temp_file, b"ok").expect("failed to create temp file for test");

    let result = validate_result_path(&temp_file.to_string_lossy());
    let canonical = temp_file
        .canonicalize()
        .expect("failed to canonicalize temp file path");

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), canonical);

    fs::remove_file(&temp_file).expect("failed to cleanup temp file");
}

#[test]
fn validate_existing_directory_path_should_return_canonical_directory_path() {
    let temp_dir = std::env::temp_dir().join(format!(
        "desktop_test_existing_dir_{}",
        next_test_id()
    ));
    fs::create_dir_all(&temp_dir).expect("failed to create temp dir for test");

    let result = validate_result_path(&temp_dir.to_string_lossy());
    let canonical = temp_dir
        .canonicalize()
        .expect("failed to canonicalize temp dir path");

    assert!(result.is_ok());
    assert_eq!(result.unwrap(), canonical);

    fs::remove_dir_all(&temp_dir).expect("failed to cleanup temp dir");
}

#[test]
fn validate_existing_file_parent_should_be_directory() {
    let temp_file = std::env::temp_dir().join(format!(
        "desktop_test_parent_dir_file_{}.txt",
        next_test_id()
    ));
    fs::write(&temp_file, b"ok").expect("failed to create temp file for test");

    let canonical = validate_result_path(&temp_file.to_string_lossy())
        .expect("expected validate_result_path to return canonical file path");

    let parent = canonical.parent().expect("file should have a parent directory");
    assert!(Path::new(parent).is_dir());

    fs::remove_file(&temp_file).expect("failed to cleanup temp file");
}
