use super::support::next_test_id;
use crate::file_mgmt::open_result_in_file_manager;

#[test]
fn open_result_in_file_manager_with_empty_path_should_return_path_is_empty_error() {
    let result = open_result_in_file_manager(" ".to_string());

    assert!(result.is_err());
    assert_eq!(result.unwrap_err(), "Path is empty");
}

#[test]
fn open_result_in_file_manager_with_missing_path_should_return_path_does_not_exist_error() {
    let missing_path = std::env::temp_dir()
        .join(format!("desktop_test_nonexistent_open_{}", next_test_id()));

    let result = open_result_in_file_manager(missing_path.to_string_lossy().to_string());

    assert!(result.is_err());
    assert!(result.unwrap_err().starts_with("Path does not exist:"));
}
