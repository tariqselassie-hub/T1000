// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! Where benchmark and training binaries write their results.
//!
//! [`write`] and [`create`] are drop-in replacements for `std::fs::write` and
//! `std::fs::File::create`. A run never touches a saved result unless asked to record:
//!
//! | mode                                    | destination of a path inside the workspace          |
//! | :-------------------------------------- | :-------------------------------------------------- |
//! | default                                 | `target/bench-scratch/<host>/<path>` (git-ignored)   |
//! | `--record` / `T1000_RECORD=1`, reference host | the path itself                                |
//! | `--record` / `T1000_RECORD=1`, other host     | `results/hosts/<host>/<path without "results/">` |
//!
//! Paths outside the workspace (an explicit `--out /tmp/x.json`) are always used as given.
//! [`write_if_better`] additionally keeps the best recorded value of a metric in a `.best`
//! sidecar and only replaces the saved file when the new value is better, unless
//! `--force` / `T1000_RECORD_FORCE=1` is set.

use std::fs::{self, File};
use std::io;
use std::path::{Component, Path, PathBuf};

use crate::hostinfo::HostProfile;

/// Which direction of a metric counts as an improvement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Better {
    Lower,
    Higher,
}

/// What [`write_if_better`] did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Outcome {
    /// The file was written (scratch mode, first record, forced, or an improvement).
    Written,
    /// An existing recorded run is at least as good; nothing was changed.
    KeptBest,
}

/// Run policy resolved from the environment and command line.
#[derive(Debug, Clone)]
pub struct Policy {
    pub record: bool,
    pub force: bool,
    pub host_slug: String,
    pub reference_host: bool,
    pub root: PathBuf,
}

fn flag(env: &str, arg: &str) -> bool {
    let on = std::env::var(env).is_ok_and(|v| matches!(v.as_str(), "1" | "true" | "yes"));
    on || std::env::args().skip(1).any(|a| a == arg)
}

fn workspace_root() -> PathBuf {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    cwd.ancestors()
        .find(|dir| {
            fs::read_to_string(dir.join("Cargo.toml")).is_ok_and(|t| t.contains("[workspace]"))
        })
        .unwrap_or(&cwd)
        .to_path_buf()
}

impl Policy {
    /// Reads `--record` / `T1000_RECORD`, `--force` / `T1000_RECORD_FORCE`, the host and the
    /// workspace root.
    pub fn detect() -> Self {
        let host = HostProfile::detect();
        Self {
            record: flag("T1000_RECORD", "--record"),
            force: flag("T1000_RECORD_FORCE", "--force"),
            host_slug: host.slug(),
            reference_host: host.is_reference_host(),
            root: workspace_root(),
        }
    }

    /// Maps the path a binary asked for to the path that is actually written.
    pub fn resolve(&self, requested: &Path) -> PathBuf {
        let abs = if requested.is_absolute() {
            requested.to_path_buf()
        } else {
            std::env::current_dir().unwrap_or_default().join(requested)
        };
        let Ok(rel) = abs.strip_prefix(&self.root) else {
            return requested.to_path_buf();
        };
        let rel = normalize(rel);
        if self.record && self.reference_host {
            self.root.join(rel)
        } else if self.record {
            let tail = rel.strip_prefix("results").unwrap_or(&rel);
            self.root.join("results/hosts").join(&self.host_slug).join(tail)
        } else {
            self.root.join("target/bench-scratch").join(&self.host_slug).join(rel)
        }
    }

    fn note(&self, requested: &Path, actual: &Path) {
        if actual == requested {
            return;
        }
        let shown = actual.strip_prefix(&self.root).unwrap_or(actual).display();
        if self.record {
            eprintln!("[results] recorded: {} -> {shown}", requested.display());
        } else {
            eprintln!(
                "[results] not recorded: wrote scratch copy {shown} instead of {} \
                 (add --record or set T1000_RECORD=1 to save)",
                requested.display()
            );
        }
    }
}

/// Collapses `.` and `..` components lexically.
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

fn prepare(path: &Path) -> io::Result<()> {
    match path.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => fs::create_dir_all(dir),
        _ => Ok(()),
    }
}

/// Drop-in for [`std::fs::write`] that follows the result-recording policy.
pub fn write(path: impl AsRef<Path>, contents: impl AsRef<[u8]>) -> io::Result<()> {
    let policy = Policy::detect();
    let actual = policy.resolve(path.as_ref());
    prepare(&actual)?;
    fs::write(&actual, contents)?;
    policy.note(path.as_ref(), &actual);
    Ok(())
}

/// Drop-in for [`std::fs::File::create`] that follows the result-recording policy.
pub fn create(path: impl AsRef<Path>) -> io::Result<File> {
    let policy = Policy::detect();
    let actual = policy.resolve(path.as_ref());
    prepare(&actual)?;
    let file = File::create(&actual)?;
    policy.note(path.as_ref(), &actual);
    Ok(file)
}

/// Writes `contents` and, when recording, keeps the file only if `value` of `metric` beats the
/// best previously recorded one (stored next to the file as `<name>.best`).
pub fn write_if_better(
    path: impl AsRef<Path>,
    contents: impl AsRef<[u8]>,
    metric: &str,
    value: f64,
    better: Better,
) -> io::Result<Outcome> {
    let policy = Policy::detect();
    write_if_better_with(&policy, path.as_ref(), contents.as_ref(), metric, value, better)
}

fn write_if_better_with(
    policy: &Policy,
    path: &Path,
    contents: &[u8],
    metric: &str,
    value: f64,
    better: Better,
) -> io::Result<Outcome> {
    let actual = policy.resolve(path);
    prepare(&actual)?;
    let sidecar = sidecar_path(&actual);
    if policy.record && !policy.force {
        if let Some((name, best)) = read_best(&sidecar) {
            if name == metric && !improves(value, best, better) {
                eprintln!("[results] not recorded: {metric} = {value} does not beat saved {best}");
                return Ok(Outcome::KeptBest);
            }
        }
    }
    fs::write(&actual, contents)?;
    if policy.record {
        fs::write(&sidecar, format!("{metric} {value:e}\n"))?;
    }
    policy.note(path, &actual);
    Ok(Outcome::Written)
}

fn improves(new: f64, best: f64, better: Better) -> bool {
    match better {
        Better::Lower => new < best,
        Better::Higher => new > best,
    }
}

fn sidecar_path(file: &Path) -> PathBuf {
    let mut name = file.file_name().unwrap_or_default().to_os_string();
    name.push(".best");
    file.with_file_name(name)
}

fn read_best(sidecar: &Path) -> Option<(String, f64)> {
    let text = fs::read_to_string(sidecar).ok()?;
    let (name, value) = text.trim().split_once(' ')?;
    Some((name.to_string(), value.parse().ok()?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(record: bool, reference: bool) -> Policy {
        Policy {
            record,
            force: false,
            host_slug: "i3-1115g4".into(),
            reference_host: reference,
            root: PathBuf::from("/ws"),
        }
    }

    #[test]
    fn default_mode_redirects_to_scratch() {
        let p = policy(false, false);
        assert_eq!(
            p.resolve(Path::new("/ws/results/experiments/a.json")),
            PathBuf::from("/ws/target/bench-scratch/i3-1115g4/results/experiments/a.json")
        );
    }

    #[test]
    fn record_on_reference_host_uses_the_requested_path() {
        let p = policy(true, true);
        assert_eq!(
            p.resolve(Path::new("/ws/results/experiments/a.json")),
            PathBuf::from("/ws/results/experiments/a.json")
        );
    }

    #[test]
    fn record_on_other_host_goes_to_host_directory() {
        let p = policy(true, false);
        assert_eq!(
            p.resolve(Path::new("/ws/results/experiments/a.json")),
            PathBuf::from("/ws/results/hosts/i3-1115g4/experiments/a.json")
        );
        assert_eq!(
            p.resolve(Path::new("/ws/tests/benchmarks/z/out.json")),
            PathBuf::from("/ws/results/hosts/i3-1115g4/tests/benchmarks/z/out.json")
        );
    }

    #[test]
    fn paths_outside_the_workspace_are_untouched() {
        for record in [false, true] {
            let p = policy(record, false);
            assert_eq!(p.resolve(Path::new("/tmp/x.json")), PathBuf::from("/tmp/x.json"));
        }
    }

    #[test]
    fn dot_dot_components_cannot_escape_the_scratch_tree() {
        let p = policy(false, false);
        let got = p.resolve(Path::new("/ws/crates/../results/a.json"));
        assert_eq!(got, PathBuf::from("/ws/target/bench-scratch/i3-1115g4/results/a.json"));
    }

    #[test]
    fn keep_best_only_replaces_with_an_improvement() {
        let root = std::env::temp_dir().join(format!("t1000_results_test_{}", std::process::id()));
        let mut p = policy(true, true);
        p.root = root.clone();
        let file = root.join("results/run.json");
        let go = |p: &Policy, body: &str, v: f64| {
            write_if_better_with(p, &file, body.as_bytes(), "latency_ms", v, Better::Lower).unwrap()
        };
        assert_eq!(go(&p, "first", 10.0), Outcome::Written);
        assert_eq!(go(&p, "slower", 12.0), Outcome::KeptBest);
        assert_eq!(fs::read_to_string(&file).unwrap(), "first");
        assert_eq!(go(&p, "faster", 8.0), Outcome::Written);
        assert_eq!(fs::read_to_string(&file).unwrap(), "faster");
        p.force = true;
        assert_eq!(go(&p, "forced", 99.0), Outcome::Written);
        assert_eq!(fs::read_to_string(&file).unwrap(), "forced");
        // Scratch mode never consults or writes the sidecar.
        let scratch = Policy { record: false, force: false, ..p.clone() };
        assert_eq!(go(&scratch, "scratch", 1000.0), Outcome::Written);
        assert_eq!(fs::read_to_string(&file).unwrap(), "forced");
        fs::remove_dir_all(&root).unwrap();
    }

    #[test]
    fn improvement_direction() {
        assert!(improves(1.0, 2.0, Better::Lower));
        assert!(!improves(2.0, 2.0, Better::Lower));
        assert!(improves(3.0, 2.0, Better::Higher));
    }
}
