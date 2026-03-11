use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Validates that a given result path exists and is accessible.
///
/// # Arguments
/// * `result_path` - The string path provided by the user or backend.
///
/// # Returns
/// A canonical `PathBuf` if valid, or an error `String` if the path is invalid or inaccessible.
pub fn validate_result_path(result_path: &str) -> Result<PathBuf, String> {
    if result_path.trim().is_empty() {
        return Err("Path is empty".to_string());
    }

    let raw_path = PathBuf::from(result_path);
    if !raw_path.exists() {
        return Err(format!("Path does not exist: {}", raw_path.display()));
    }

    let canonical_path = raw_path.canonicalize().map_err(|e| {
        if e.kind() == std::io::ErrorKind::PermissionDenied {
            format!(
                "Permission denied when resolving path: {}",
                raw_path.display()
            )
        } else {
            format!("Failed to resolve path {}: {e}", raw_path.display())
        }
    })?;

    if canonical_path.is_file() {
        fs::File::open(&canonical_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                format!(
                    "Permission denied when accessing file: {}",
                    canonical_path.display()
                )
            } else {
                format!("Cannot access file {}: {e}", canonical_path.display())
            }
        })?;
    } else if canonical_path.is_dir() {
        fs::read_dir(&canonical_path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::PermissionDenied {
                format!(
                    "Permission denied when accessing directory: {}",
                    canonical_path.display()
                )
            } else {
                format!("Cannot access directory {}: {e}", canonical_path.display())
            }
        })?;
    } else {
        return Err(format!(
            "Path is neither a regular file nor directory: {}",
            canonical_path.display()
        ));
    }

    Ok(canonical_path)
}

/// Opens the system's default file manager at a specific path.
///
/// Uses platform-specific commands:
/// - Windows: `explorer /select, <path>` or `explorer <path>`
/// - macOS: `open -R <path>` or `open <path>`
/// - Linux: `xdg-open <path>`
///
/// # Arguments
/// * `path` - The path to reveal or open.
pub fn open_in_file_manager(path: &Path) -> Result<(), String> {
    #[cfg(target_os = "windows")]
    {
        let status = if path.is_file() {
            Command::new("explorer").arg("/select,").arg(path).status()
        } else {
            Command::new("explorer").arg(path).status()
        }
        .map_err(|e| format!("Failed to launch Explorer: {e}"))?;

        if status.success() {
            return Ok(());
        }
        return Err("Explorer returned a non-zero exit code".to_string());
    }

    #[cfg(target_os = "macos")]
    {
        let status = if path.is_file() {
            Command::new("open").arg("-R").arg(path).status()
        } else {
            Command::new("open").arg(path).status()
        }
        .map_err(|e| format!("Failed to launch Finder: {e}"))?;

        if status.success() {
            return Ok(());
        }
        return Err("Finder returned a non-zero exit code".to_string());
    }

    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let target = if path.is_file() {
            path.parent()
                .ok_or_else(|| "File has no parent directory to open".to_string())?
        } else {
            path
        };

        let status = Command::new("xdg-open")
            .arg(target)
            .status()
            .map_err(|e| format!("Failed to launch file manager (xdg-open): {e}"))?;

        if status.success() {
            return Ok(());
        }
        return Err("xdg-open returned a non-zero exit code".to_string());
    }

    #[allow(unreachable_code)]
    Err("Unsupported operating system for opening file manager".to_string())
}

/// Tauri command handler to open a result path in the file manager.
#[tauri::command]
pub fn open_result_in_file_manager(result_path: String) -> Result<(), String> {
    let canonical_path = validate_result_path(&result_path)?;

    open_in_file_manager(&canonical_path)
}
