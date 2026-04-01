#[deprecated = r"Switching to native inference mode in Rust, so this Python subprocess management code is no longer used. 
    Keeping it around for now in case we need to reference it for the native implementation, 
    but it will likely be removed in the future."]
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

impl BackendProcess {
    /// Splits the process into the child handle and owned stdio streams.
    #[must_use]
    pub fn into_parts(self) -> (Child, ChildStdin, BufReader<ChildStdout>) {
        (self.child, self.stdin, self.stdout)
    }
}

/// Spawns a new backend process with proper IO redirection and environment setup.
///
/// # Arguments
/// * `launch` - The command configuration to execute.
/// * `repo_root` - Optional repository root used to set `PYTHONPATH` for development.
///
/// # Errors
/// Returns an error when the child process cannot be spawned or its stdio
/// handles cannot be captured.
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
    }

    let mut child = launch_cmd.spawn().map_err(|e| {
        format!("Failed to spawn backend process ({}): {e}", launch.program)
    })?;

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

// Repository-root inference via Python script is no longer required when
// launching the bundled backend binary. Development-time helpers were
// removed to simplify launch logic.

// `PYTHONPATH` manipulation removed: production runtime only launches the
// bundled backend binary. Development-time Python IPC setup was intentionally
// removed to avoid touching Python dev layout from production code paths.

/// Loosely loads environment variables from a `.env` file if present near the executable.
/// This mimics `python-dotenv` behavior for desktop dev environments.
pub fn load_desktop_env(cwd: &Path, exe_dir: &Path) {
    let env_suffixes = [
        ".env",
        "app/gui/workbench/.env",
        "../app/gui/workbench/.env",
        "../../app/gui/workbench/.env",
        "../../../app/gui/workbench/.env",
    ];

    let env_path = first_existing_path(cwd, &env_suffixes).or_else(|| {
        first_existing_path(
            exe_dir,
            &["../.env", "../../.env", "../../../.env"],
        )
    });

    let Some(path) = env_path else {
        return;
    };

    let Ok(contents) = fs::read_to_string(path) else {
        return;
    };

    for raw_line in contents.lines() {
        let line = raw_line.trim();
        if !(line.is_empty() || line.starts_with('#'))
            && let Some((key, value)) = line.split_once('=')
        {
            let env_key = key.trim();
            if !(env_key.is_empty() || std::env::var(env_key).is_ok()) {
                let env_value =
                    value.trim().trim_matches('"').trim_matches('\'');
                if !env_value.is_empty() {
                    // Note: std::env::set_var is unsafe in Rust 2024 and forbidden by project policy.
                    // We skip loading into the current process's environment.
                    // Future improvement: Store in a local Map for child process spawning.
                }
            }
        }
    }
}

/// Locates the bundled backend binary (e.g. created by `PyInstaller`).
///
/// # Errors
/// Returns an error when `app/gui/workbench/backend/main` cannot be found from
/// either the current working directory or the executable directory.
pub fn resolve_backend_binary_path_from(
    cwd: &Path,
    exe_dir: &Path,
) -> Result<PathBuf, String> {
    if let Ok(override_path) = std::env::var("FASTSURFER_BACKEND_BIN") {
        let p = PathBuf::from(override_path);
        if p.exists() {
            return std::fs::canonicalize(p).map_err(|e| {
                format!("failed to canonicalize override path: {e}")
            });
        }
    }

    let common_suffixes = [
        "backend/main",
        "app/gui/workbench/backend/main",
        "gui/workbench/backend/main",
        "resources/backend/main",
        "../backend/main",
    ];

    #[cfg(windows)]
    let common_suffixes_win = [
        "backend/main.exe",
        "app/gui/workbench/backend/main.exe",
        "gui/workbench/backend/main.exe",
        "resources/backend/main.exe",
        "../backend/main.exe",
    ];

    // Build candidate base directories in order: exe parent (packaged), exe_dir, cwd (repo).
    let mut bases: Vec<PathBuf> = Vec::new();
    if let Ok(current_exe) = std::env::current_exe()
        && let Some(parent) = current_exe.parent()
    {
        bases.push(parent.to_path_buf());
    }
    bases.push(exe_dir.to_path_buf());
    bases.push(cwd.to_path_buf());

    for base in &bases {
        if let Some(found) =
            first_existing_path(base.as_path(), &common_suffixes)
        {
            return std::fs::canonicalize(found).map_err(|e| {
                format!("failed to canonicalize backend path: {e}")
            });
        }
        #[cfg(windows)]
        {
            if let Some(found) =
                first_existing_path(base.as_path(), &common_suffixes_win)
            {
                return std::fs::canonicalize(found).map_err(|e| {
                    format!("failed to canonicalize backend path: {e}")
                });
            }
        }
    }

    Err("Could not find bundled backend binary. Use FASTSURFER_BACKEND_BIN to override.".to_string())
}

/// Determines the best command to launch the backend: either the bundled binary or python script.
///
/// # Errors
/// Returns an error when neither a bundled backend binary nor a usable Python
/// runtime with the development IPC script can be found.
pub fn resolve_backend_launch_command_from(
    cwd: &Path,
    exe_dir: &Path,
) -> Result<BackendLaunchCommand, String> {
    // Only allow launching the bundled backend binary.
    let binary_path = resolve_backend_binary_path_from(cwd, exe_dir)?;

    Ok(BackendLaunchCommand {
        program: binary_path.to_string_lossy().to_string(),
        args: Vec::new(),
    })
}
