use std::env;
use std::ffi::OsStr;
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=FASTSURFER_ORT_RUNTIME_DIR");
    println!("cargo:rerun-if-env-changed=FASTSURFER_ORT_DYLIB_PATH");
    println!("cargo:rerun-if-env-changed=FASTSURFER_ORT_DOWNLOAD_DIR");
    println!("cargo:rerun-if-env-changed=FASTSURFER_ORT_VERSION");
    println!("cargo:rerun-if-env-changed=FASTSURFER_ORT_DOWNLOAD_URL");

    if let Err(error) = stage_ort_runtime() {
        panic!("ORT runtime staging failed: {error}");
    }

    tauri_build::build();
}

fn stage_ort_runtime() -> Result<(), String> {
    let runtime_dir = resolve_ort_runtime_dir()?;

    if !runtime_dir.exists() || !runtime_dir.is_dir() {
        return Err(format!(
            "ORT runtime directory does not exist or is not a directory: {}",
            runtime_dir.display()
        ));
    }

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").map_err(|e| e.to_string())?);
    let out_dir = manifest_dir
        .join("resources")
        .join("ort")
        .join(platform_folder());

    if runtime_dir == out_dir {
        if dir_contains_runtime_files(&out_dir)? {
            println!(
                "cargo:warning=using existing staged ORT runtime in '{}'",
                out_dir.display()
            );
            return Ok(());
        }
        return Err(format!(
            "ORT runtime source equals staging dir '{}' but no runtime files were found",
            out_dir.display()
        ));
    }

    if out_dir.exists() {
        fs::remove_dir_all(&out_dir).map_err(|e| {
            format!(
                "failed to clear staged ORT dir '{}': {e}",
                out_dir.display()
            )
        })?;
    }
    fs::create_dir_all(&out_dir).map_err(|e| {
        format!(
            "failed to create staged ORT dir '{}': {e}",
            out_dir.display()
        )
    })?;

    let mut copied = 0usize;
    let entries = fs::read_dir(&runtime_dir).map_err(|e| {
        format!(
            "failed to read ORT runtime dir '{}': {e}",
            runtime_dir.display()
        )
    })?;

    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }

        let Some(file_name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };

        if !is_ort_runtime_file(file_name) {
            continue;
        }

        let target = out_dir.join(file_name);
        fs::copy(&path, &target).map_err(|e| {
            format!(
                "failed to copy ORT runtime file '{}' -> '{}': {e}",
                path.display(),
                target.display()
            )
        })?;
        copied += 1;
    }

    if copied == 0 {
        return Err(format!(
            "no ORT runtime files matching platform were found in '{}'. Expected files like {}",
            runtime_dir.display(),
            expected_runtime_hint()
        ));
    }
    println!(
        "cargo:warning=staged {copied} ORT runtime file(s) from '{}' into '{}'",
        runtime_dir.display(),
        out_dir.display()
    );

    Ok(())
}

fn resolve_ort_runtime_dir() -> Result<PathBuf, String> {
    let manifest_dir = PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").map_err(|e| format!("missing CARGO_MANIFEST_DIR: {e}"))?,
    );

    if let Some(explicit) = env::var_os("FASTSURFER_ORT_RUNTIME_DIR") {
        let path = PathBuf::from(explicit);
        if dir_contains_runtime_files(&path)? {
            return Ok(path);
        }
        return Err(format!(
            "FASTSURFER_ORT_RUNTIME_DIR is set but no runtime files were found in '{}'. Expected {}",
            path.display(),
            expected_runtime_hint()
        ));
    }

    if let Some(explicit_file) = env::var_os("FASTSURFER_ORT_DYLIB_PATH") {
        let file = PathBuf::from(explicit_file);
        if let Some(parent) = file.parent() {
            if file.exists() && file.is_file() {
                return Ok(parent.to_path_buf());
            }
            return Err(format!(
                "FASTSURFER_ORT_DYLIB_PATH points to missing file: {}",
                file.display()
            ));
        }
        return Err(format!(
            "FASTSURFER_ORT_DYLIB_PATH is set but has no parent directory: {}",
            file.display()
        ));
    }

    let bundled = manifest_dir
        .join("resources")
        .join("ort")
        .join(platform_folder());
    if dir_contains_runtime_files(&bundled)? {
        return Ok(bundled);
    }

    let download_root = env::var_os("FASTSURFER_ORT_DOWNLOAD_DIR")
        .map_or_else(|| manifest_dir.join(".ort-runtime-cache"), PathBuf::from);

    let runtime_dir = download_ort_runtime_if_missing(&download_root)?;
    if dir_contains_runtime_files(&runtime_dir)? {
        return Ok(runtime_dir);
    }

    Err(format!(
        "unable to resolve ORT runtime. Set FASTSURFER_ORT_RUNTIME_DIR/FASTSURFER_ORT_DYLIB_PATH, or allow auto-download to '{}'. Expected {}",
        download_root.display(),
        expected_runtime_hint()
    ))
}

fn dir_contains_runtime_files(dir: &std::path::Path) -> Result<bool, String> {
    if !dir.exists() || !dir.is_dir() {
        return Ok(false);
    }

    let entries = fs::read_dir(dir)
        .map_err(|e| format!("failed to read runtime dir '{}': {e}", dir.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(name) = path.file_name().and_then(OsStr::to_str) else {
            continue;
        };
        if is_ort_runtime_file(name) {
            return Ok(true);
        }
    }

    Ok(false)
}

fn download_ort_runtime_if_missing(download_root: &std::path::Path) -> Result<PathBuf, String> {
    let version = env::var("FASTSURFER_ORT_VERSION").unwrap_or_else(|_| "1.23.2".to_string());
    let platform = platform_folder();
    let runtime_dir = download_root.join(platform).join(&version).join("runtime");

    if dir_contains_runtime_files(&runtime_dir)? {
        println!(
            "cargo:warning=using cached ORT runtime from '{}'",
            runtime_dir.display()
        );
        return Ok(runtime_dir);
    }

    fs::create_dir_all(&runtime_dir).map_err(|e| {
        format!(
            "failed to create ORT runtime download dir '{}': {e}",
            runtime_dir.display()
        )
    })?;

    let (file_name, default_url, is_zip) = ort_download_spec(&version);
    let url = env::var("FASTSURFER_ORT_DOWNLOAD_URL").unwrap_or(default_url);
    let archive_path = download_root.join(platform).join(&version).join(file_name);

    if let Some(parent) = archive_path.parent() {
        fs::create_dir_all(parent).map_err(|e| {
            format!(
                "failed to create ORT archive parent dir '{}': {e}",
                parent.display()
            )
        })?;
    }

    println!("cargo:warning=downloading ORT runtime from {url}");
    let curl_status = Command::new("curl")
        .args(["-L", "--fail", "--retry", "3", "-o"])
        .arg(&archive_path)
        .arg(&url)
        .status()
        .map_err(|e| format!("failed to execute curl for ORT download: {e}"))?;

    if !curl_status.success() {
        return Err(format!(
            "failed to download ORT runtime from '{url}' (curl exit: {curl_status})"
        ));
    }

    if is_zip {
        let unzip_status = Command::new("unzip")
            .arg("-o")
            .arg(&archive_path)
            .arg("-d")
            .arg(&runtime_dir)
            .status()
            .map_err(|e| format!("failed to execute unzip: {e}"))?;
        if !unzip_status.success() {
            return Err(format!(
                "failed to extract ORT archive '{}' with unzip",
                archive_path.display()
            ));
        }
    } else {
        let tar_status = Command::new("tar")
            .arg("-xzf")
            .arg(&archive_path)
            .arg("-C")
            .arg(&runtime_dir)
            .status()
            .map_err(|e| format!("failed to execute tar: {e}"))?;
        if !tar_status.success() {
            return Err(format!(
                "failed to extract ORT archive '{}' with tar",
                archive_path.display()
            ));
        }
    }

    if let Some(found) = find_runtime_dir_with_libs(&runtime_dir)? {
        return Ok(found);
    }

    Err(format!(
        "ORT download succeeded but runtime libraries were not found under '{}'",
        runtime_dir.display()
    ))
}

fn find_runtime_dir_with_libs(root: &std::path::Path) -> Result<Option<PathBuf>, String> {
    if dir_contains_runtime_files(root)? {
        return Ok(Some(root.to_path_buf()));
    }

    let entries = fs::read_dir(root)
        .map_err(|e| format!("failed to scan ORT extracted dir '{}': {e}", root.display()))?;
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        if dir_contains_runtime_files(&path)? {
            return Ok(Some(path));
        }

        let nested = fs::read_dir(&path)
            .map_err(|e| format!("failed to scan nested ORT dir '{}': {e}", path.display()))?;
        for nested_entry in nested.flatten() {
            let nested_path = nested_entry.path();
            if nested_path.is_dir() && dir_contains_runtime_files(&nested_path)? {
                return Ok(Some(nested_path));
            }
        }
    }

    Ok(None)
}

fn ort_download_spec(version: &str) -> (&'static str, String, bool) {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        let file_name = "onnxruntime-linux-x64.tgz";
        let url = format!(
            "https://github.com/microsoft/onnxruntime/releases/download/v{version}/onnxruntime-linux-x64-{version}.tgz"
        );
        (file_name, url, false)
    }

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    {
        let file_name = "onnxruntime-osx-x86_64.tgz";
        let url = format!(
            "https://github.com/microsoft/onnxruntime/releases/download/v{version}/onnxruntime-osx-x86_64-{version}.tgz"
        );
        (file_name, url, false)
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        let file_name = "onnxruntime-osx-arm64.tgz";
        let url = format!(
            "https://github.com/microsoft/onnxruntime/releases/download/v{version}/onnxruntime-osx-arm64-{version}.tgz"
        );
        (file_name, url, false)
    }

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        let file_name = "onnxruntime-win-x64.zip";
        let url = format!(
            "https://github.com/microsoft/onnxruntime/releases/download/v{version}/onnxruntime-win-x64-{version}.zip"
        );
        (file_name, url, true)
    }

    #[cfg(not(any(
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "x86_64"),
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "windows", target_arch = "x86_64")
    )))]
    {
        panic!("automatic ORT runtime download is not configured for this target platform/arch")
    }
}

fn expected_runtime_hint() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "onnxruntime*.dll"
    }

    #[cfg(target_os = "linux")]
    {
        "libonnxruntime*.so*"
    }

    #[cfg(target_os = "macos")]
    {
        "libonnxruntime*.dylib"
    }
}

fn platform_folder() -> &'static str {
    #[cfg(target_os = "windows")]
    {
        "windows"
    }

    #[cfg(target_os = "linux")]
    {
        "linux"
    }

    #[cfg(target_os = "macos")]
    {
        "macos"
    }
}

fn is_ort_runtime_file(file_name: &str) -> bool {
    #[cfg(target_os = "windows")]
    {
        let lower = file_name.to_ascii_lowercase();
        lower.starts_with("onnxruntime") && lower.ends_with(".dll")
    }

    #[cfg(target_os = "linux")]
    {
        file_name.starts_with("libonnxruntime") && file_name.contains(".so")
    }

    #[cfg(target_os = "macos")]
    {
        file_name.starts_with("libonnxruntime") && file_name.ends_with(".dylib")
    }
}
