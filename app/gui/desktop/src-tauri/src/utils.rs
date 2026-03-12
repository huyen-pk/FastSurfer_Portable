use std::path::{Path, PathBuf};

/// Finds the first existing path from a list of relative suffixes joined to a base path.
///
/// # Arguments
/// * `base` - The base directory to search from.
/// * `suffixes` - A list of relative paths to check for existence.
///
/// # Returns
/// An `Option<PathBuf>` containing the first valid path found, or `None` if none exist.
#[must_use]
pub fn first_existing_path(base: &Path, suffixes: &[&str]) -> Option<PathBuf> {
    suffixes
        .iter()
        .map(|suffix| base.join(suffix))
        .find(|candidate| candidate.exists())
}
