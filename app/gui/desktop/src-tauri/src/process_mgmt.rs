#[deprecated = r#"Switching to native inference mode in Rust, so this Python subprocess management code is no longer used. 
    Keeping it around for now in case we need to reference it for the native implementation, 
    but it will likely be removed in the future."#]
use crate::utils::first_existing_path;
use std::fs;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

/// Configuration for launching a backend subprocess.
#[derive(Clone)]
pub struct BackendLaunchCommand {
    /// The program/executable name or path.
    pub program: String,
    /// Arguments to pass to the program.
    pub args: Vec<String>,
}

/// Represents a running backend process with its I/O streams.
pub struct BackendProcess {
    /// The child process handle.
    pub child: Child,
    /// The standard input stream for sending commands.
    pub stdin: ChildStdin,
    /// The buffered standard output stream for receiving responses.
    pub stdout: BufReader<ChildStdout>,
}

/// Spawns a new backend process with proper IO redirection and environment setup.
///
/// # Arguments
/// * `launch` - The command configuration to execute.
/// * `repo_root` - Optional repository root used to set `PYTHONPATH` for development.
///
/// # Returns
/// A `BackendProcess` handle or an error string.
pub fn spawn_backend_process(
    launch: &BackendLaunchCommand,
    repo_root: Option<&PathBuf>,
) -> Result<BackendProcess, String> {
    let mut launch_cmd = Command::new(&launch.program);
    launch_cmd
        .args(&launch.args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit());

    if launch.args.is_empty() {
        let sidecar_dir = Path::new(&launch.program)
            .parent()
            .ok_or_else(|| "Cannot resolve sidecar directory".to_string())?;
        launch_cmd.current_dir(sidecar_dir);
    } else if let Some(root) = repo_root {
        launch_cmd.current_dir(root);
        launch_cmd.env("PYTHONPATH", with_repo_on_pythonpath(root));
    }

    let mut child = launch_cmd
        .spawn()
        .map_err(|e| format!("Failed to spawn backend process ({}): {e}", launch.program))?;

    let stdin = child
        .stdin
        .take()
        .ok_or_else(|| "Failed to capture backend stdin".to_string())?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "Failed to capture backend stdout".to_string())?;

    Ok(BackendProcess {
        child,
        stdin,
        stdout: BufReader::new(stdout),
    })
}

/// Attempts to infer the repository root by walking up from the python script path.
/// Looks for the `FastSurferCNN` directory as a marker.
pub fn resolve_repo_root_from_python_script(script_path: &Path) -> Option<PathBuf> {
    script_path
        .ancestors()
        .find(|ancestor| ancestor.join("FastSurferCNN").exists())
        .map(Path::to_path_buf)
}

/// Helper to prepend the repo root to the existing PYTHONPATH.
fn with_repo_on_pythonpath(repo_root: &Path) -> String {
    let repo_root_str = repo_root.to_string_lossy().to_string();
    if let Ok(existing) = std::env::var("PYTHONPATH") {
        if existing
            .split(':')
            .any(|entry| entry.trim() == repo_root_str)
        {
            return existing;
        }
        if existing.trim().is_empty() {
            return repo_root_str;
        }
        return format!("{repo_root_str}:{existing}");
    }
    repo_root_str
}

/// Loosely loads environment variables from a `.env` file if present near the executable.
/// This mimics `python-dotenv` behavior for desktop dev environments.
pub fn load_desktop_env(cwd: &Path, exe_dir: &Path) {
    let env_suffixes = [
        ".env",
        "app/gui/desktop/.env",
        "../app/gui/desktop/.env",
        "../../app/gui/desktop/.env",
        "../../../app/gui/desktop/.env",
    ];

    let env_path = first_existing_path(cwd, &env_suffixes)
        .or_else(|| first_existing_path(exe_dir, &["../.env", "../../.env", "../../../.env"]));

    let Some(path) = env_path else {
        return;
    };

    let Ok(contents) = fs::read_to_string(path) else {
        return;
    };

    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        let Some((key, value)) = line.split_once('=') else {
            continue;
        };

        let env_key = key.trim();
        if env_key.is_empty() || std::env::var(env_key).is_ok() {
            continue;
        }

        let env_value = value.trim().trim_matches('"').trim_matches('\'');
        if env_value.is_empty() {
            continue;
        }

        // Note: std::env::set_var is unsafe in Rust 2024 and forbidden by project policy.
        // We skip loading into the current process's environment.
        // Future improvement: Store in a local Map for child process spawning.
    }
}

/// Locates the bundled backend binary (e.g. created by PyInstaller).
pub fn resolve_backend_binary_path_from(cwd: &Path, exe_dir: &Path) -> Result<PathBuf, String> {
    let cwd_suffixes = [
        "app/gui/desktop/backend/main",
        "gui/desktop/backend/main",
        "backend/main",
        "../backend/main",
        "../../backend/main",
        "../../../backend/main",
    ];

    let exe_suffixes = [
        "../backend/main",
        "../../backend/main",
        "../../../backend/main",
        "../../../../backend/main",
        "../../../../../backend/main",
        "../../../../../../backend/main",
    ];

    first_existing_path(cwd, &cwd_suffixes)
        .or_else(|| first_existing_path(exe_dir, &exe_suffixes))
        .ok_or_else(|| {
            "Could not find bundled backend binary app/gui/desktop/backend/main".to_string()
        })
}

/// Locates the python backend script `ipc_server.py` for development mode.
pub fn resolve_python_backend_script_path_from(cwd: &Path, exe_dir: &Path) -> Option<PathBuf> {
    let cwd_suffixes = [
        "app/backend/ipc_server.py",
        "../app/backend/ipc_server.py",
        "../../app/backend/ipc_server.py",
        "../../../app/backend/ipc_server.py",
        "backend/ipc_server.py",
        "../backend/ipc_server.py",
        "../../backend/ipc_server.py",
        "../../../backend/ipc_server.py",
    ];

    let exe_suffixes = [
        "../app/backend/ipc_server.py",
        "../../app/backend/ipc_server.py",
        "../../../app/backend/ipc_server.py",
        "../../../../app/backend/ipc_server.py",
    ];

    first_existing_path(cwd, &cwd_suffixes).or_else(|| first_existing_path(exe_dir, &exe_suffixes))
}

/// Finds a valid python executable, checking environment overrides first.
pub fn resolve_python_executable() -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();

    if let Ok(py_bin) = std::env::var("FASTSURFER_PYTHON_BIN")
        && !py_bin.trim().is_empty()
    {
        candidates.push(py_bin);
    }

    if let Ok(py_bin) = std::env::var("PYTHON_BIN")
        && !py_bin.trim().is_empty()
    {
        candidates.push(py_bin);
    }

    candidates.push("python3".to_string());
    candidates.push("python".to_string());

    candidates.into_iter().find(|candidate| {
        Command::new(candidate)
            .arg("--version")
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
    })
}

/// Determines the best command to launch the backend: either the bundled binary or python script.
pub fn resolve_backend_launch_command_from(
    cwd: &Path,
    exe_dir: &Path,
) -> Result<BackendLaunchCommand, String> {
    if let Ok(binary_path) = resolve_backend_binary_path_from(cwd, exe_dir) {
        return Ok(BackendLaunchCommand {
            program: binary_path.to_string_lossy().to_string(),
            args: Vec::new(),
        });
    }

    let python_script = resolve_python_backend_script_path_from(cwd, exe_dir);
    let python_exec = resolve_python_executable();

    if let (Some(script), Some(py)) = (python_script, python_exec) {
        return Ok(BackendLaunchCommand {
            program: py,
            args: vec![script.to_string_lossy().to_string()],
        });
    }

    Err(
        "Could not resolve backend launch command: bundled backend binary not found and no Python interpreter available. Install Python or set FASTSURFER_PYTHON_BIN, or provide bundled backend binary app/gui/desktop/backend/main.".to_string(),
    )
}
