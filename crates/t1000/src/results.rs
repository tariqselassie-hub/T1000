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
//!
//! [`write_scored`] adds keep-best recording. A `.best` sidecar stores the settings
//! fingerprint, the deciding metric and an optional guard metric of the saved run. A new run
//! replaces it only if the metric improves and the guard has not regressed by more than its
//! tolerance. A run with different settings never competes with the saved one: it is saved
//! beside it as `<stem>.<settings-hash>.<ext>` with its own sidecar. `--force` /
//! `T1000_RECORD_FORCE=1` replaces the saved run unconditionally. [`write_if_better`] is the
//! single-metric form.

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
    /// Recorded: first record, forced, or an improvement.
    Written,
    /// Not recording: the output went to the scratch directory and saved results are untouched.
    Scratch,
    /// The settings differ from the saved run, so the result was saved as a sibling variant.
    WrittenVariant,
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

/// A secondary metric that must not regress by more than `tolerance` (in its own units).
#[derive(Debug, Clone, Copy)]
pub struct Guard<'a> {
    pub name: &'a str,
    pub value: f64,
    pub better: Better,
    pub tolerance: f64,
}

/// What a run is judged on when recording.
#[derive(Debug, Clone, Copy)]
pub struct Score<'a> {
    /// The deciding metric (for example held-out loss).
    pub metric: &'a str,
    pub value: f64,
    pub better: Better,
    pub guard: Option<Guard<'a>>,
    /// Fingerprint of everything that makes two runs comparable (sizes, epochs, rates).
    pub settings: &'a str,
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
    let score = Score { metric, value, better, guard: None, settings: "" };
    write_scored(path, contents, &score)
}

/// Writes `contents` under the keep-best rules described in the module docs.
pub fn write_scored(
    path: impl AsRef<Path>,
    contents: impl AsRef<[u8]>,
    score: &Score,
) -> io::Result<Outcome> {
    let policy = Policy::detect();
    write_scored_with(&policy, path.as_ref(), contents.as_ref(), score)
}

fn write_scored_with(
    policy: &Policy,
    path: &Path,
    contents: &[u8],
    score: &Score,
) -> io::Result<Outcome> {
    let actual = policy.resolve(path);
    prepare(&actual)?;
    if !policy.record {
        fs::write(&actual, contents)?;
        policy.note(path, &actual);
        return Ok(if actual == path { Outcome::Written } else { Outcome::Scratch });
    }
    let mut target = actual.clone();
    let mut variant = false;
    if !policy.force {
        if let Some(saved) = read_saved(&sidecar_path(&target)) {
            if saved.settings != score.settings {
                target = variant_path(&actual, score.settings);
                variant = true;
            }
        }
        if let Some(saved) = read_saved(&sidecar_path(&target)) {
            if let Some(reason) = rejection(&saved, score) {
                eprintln!("[results] not recorded: {reason}");
                return Ok(Outcome::KeptBest);
            }
        }
    }
    fs::write(&target, contents)?;
    fs::write(sidecar_path(&target), render(score))?;
    policy.note(path, &target);
    Ok(if variant { Outcome::WrittenVariant } else { Outcome::Written })
}

/// Why `score` must not replace `saved`, if it must not.
fn rejection(saved: &Saved, score: &Score) -> Option<String> {
    if saved.metric != score.metric {
        return None;
    }
    if !improves(score.value, saved.value, score.better) {
        return Some(format!(
            "{} = {} does not beat saved {}",
            score.metric, score.value, saved.value
        ));
    }
    let (g, (saved_name, saved_value)) = (score.guard?, saved.guard.as_ref()?);
    if g.name != saved_name.as_str() {
        return None;
    }
    let regressed = match g.better {
        Better::Higher => g.value < saved_value - g.tolerance,
        Better::Lower => g.value > saved_value + g.tolerance,
    };
    regressed.then(|| {
        format!(
            "{} improved but {} = {} regressed past saved {} (tolerance {})",
            score.metric, g.name, g.value, saved_value, g.tolerance
        )
    })
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

/// `<stem>.<8 hex digits of the settings hash>.<ext>` next to `file`.
///
/// Companion files of a result (a CSV next to a JSON) use this when [`Outcome::WrittenVariant`]
/// was returned, so both land under the same variant name.
pub fn variant_path(file: &Path, settings: &str) -> PathBuf {
    let hash = settings
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3));
    let stem = file.file_stem().unwrap_or_default().to_string_lossy();
    let name = match file.extension() {
        Some(ext) => format!("{stem}.{:08x}.{}", hash as u32, ext.to_string_lossy()),
        None => format!("{stem}.{:08x}", hash as u32),
    };
    file.with_file_name(name)
}

/// The recorded best of a saved run.
#[derive(Debug, Clone, PartialEq)]
struct Saved {
    settings: String,
    metric: String,
    value: f64,
    guard: Option<(String, f64)>,
}

fn render(score: &Score) -> String {
    let mut out = format!("settings {}\nmetric {} {:e}\n", score.settings, score.metric, score.value);
    if let Some(g) = score.guard {
        out.push_str(&format!("guard {} {:e}\n", g.name, g.value));
    }
    out
}

/// Reads the current format, or the earlier single-line `<metric> <value>` form.
fn read_saved(sidecar: &Path) -> Option<Saved> {
    let text = fs::read_to_string(sidecar).ok()?;
    let mut saved = Saved { settings: String::new(), metric: String::new(), value: 0.0, guard: None };
    let pair = |rest: &str| -> Option<(String, f64)> {
        let (name, v) = rest.trim().rsplit_once(' ')?;
        Some((name.to_string(), v.parse().ok()?))
    };
    let mut lines = text.lines().peekable();
    if lines.peek().is_some_and(|l| !l.starts_with("settings") && !l.starts_with("metric")) {
        let (metric, value) = pair(lines.next()?)?;
        return Some(Saved { metric, value, ..saved });
    }
    for line in lines {
        match line.split_once(' ') {
            Some(("settings", rest)) => saved.settings = rest.to_string(),
            Some(("metric", rest)) => (saved.metric, saved.value) = pair(rest)?,
            Some(("guard", rest)) => saved.guard = pair(rest),
            _ => {}
        }
    }
    (!saved.metric.is_empty()).then_some(saved)
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

    fn temp_policy(tag: &str) -> Policy {
        let root = std::env::temp_dir().join(format!("t1000_results_{tag}_{}", std::process::id()));
        Policy { root, ..policy(true, true) }
    }

    fn score<'a>(value: f64, guard: Option<f64>, settings: &'a str) -> Score<'a> {
        Score {
            metric: "holdout_loss",
            value,
            better: Better::Lower,
            guard: guard.map(|v| Guard { name: "accuracy", value: v, better: Better::Higher, tolerance: 1.0 }),
            settings,
        }
    }

    fn put(p: &Policy, file: &Path, body: &str, s: &Score) -> Outcome {
        write_scored_with(p, file, body.as_bytes(), s).unwrap()
    }

    #[test]
    fn keep_best_only_replaces_with_an_improvement() {
        let p = temp_policy("best");
        let file = p.root.join("results/run.json");
        assert_eq!(put(&p, &file, "first", &score(10.0, None, "a")), Outcome::Written);
        assert_eq!(put(&p, &file, "slower", &score(12.0, None, "a")), Outcome::KeptBest);
        assert_eq!(fs::read_to_string(&file).unwrap(), "first");
        assert_eq!(put(&p, &file, "faster", &score(8.0, None, "a")), Outcome::Written);
        assert_eq!(fs::read_to_string(&file).unwrap(), "faster");
        let forced = Policy { force: true, ..p.clone() };
        assert_eq!(put(&forced, &file, "forced", &score(99.0, None, "a")), Outcome::Written);
        assert_eq!(fs::read_to_string(&file).unwrap(), "forced");
        let scratch = Policy { record: false, ..p.clone() };
        assert_eq!(put(&scratch, &file, "scratch", &score(1000.0, None, "a")), Outcome::Scratch);
        assert_eq!(fs::read_to_string(&file).unwrap(), "forced");
        fs::remove_dir_all(&p.root).unwrap();
    }

    #[test]
    fn guard_blocks_a_loss_win_that_costs_accuracy() {
        let p = temp_policy("guard");
        let file = p.root.join("results/run.json");
        put(&p, &file, "base", &score(0.50, Some(90.0), "a"));
        // Better loss, accuracy down 5 points: rejected.
        assert_eq!(put(&p, &file, "overfit", &score(0.40, Some(85.0), "a")), Outcome::KeptBest);
        // Better loss, accuracy down within the 1 point tolerance: accepted.
        assert_eq!(put(&p, &file, "ok", &score(0.40, Some(89.5), "a")), Outcome::Written);
        assert_eq!(fs::read_to_string(&file).unwrap(), "ok");
        // Worse loss with better accuracy: still rejected, loss decides.
        assert_eq!(put(&p, &file, "lucky", &score(0.45, Some(99.0), "a")), Outcome::KeptBest);
        fs::remove_dir_all(&p.root).unwrap();
    }

    #[test]
    fn different_settings_never_replace_the_saved_run() {
        let p = temp_policy("variant");
        let file = p.root.join("results/run.json");
        put(&p, &file, "owner", &score(0.50, None, "epochs=20"));
        let short = score(0.10, None, "epochs=3");
        assert_eq!(put(&p, &file, "short", &short), Outcome::WrittenVariant);
        assert_eq!(fs::read_to_string(&file).unwrap(), "owner");
        let variant = variant_path(&file, "epochs=3");
        assert_eq!(fs::read_to_string(&variant).unwrap(), "short");
        // The variant has its own best: a worse run with the same settings is rejected.
        assert_eq!(put(&p, &file, "worse", &score(0.30, None, "epochs=3")), Outcome::KeptBest);
        assert_eq!(fs::read_to_string(&variant).unwrap(), "short");
        fs::remove_dir_all(&p.root).unwrap();
    }

    #[test]
    fn legacy_single_line_sidecar_is_understood() {
        let p = temp_policy("legacy");
        let file = p.root.join("results/run.json");
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, "saved").unwrap();
        fs::write(sidecar_path(&file), "latency_ms 1.0e4\n").unwrap();
        let s = Score { metric: "latency_ms", value: 12000.0, better: Better::Lower, guard: None, settings: "" };
        assert_eq!(put(&p, &file, "slower", &s), Outcome::KeptBest);
        assert_eq!(fs::read_to_string(&file).unwrap(), "saved");
        fs::remove_dir_all(&p.root).unwrap();
    }

    #[test]
    fn improvement_direction() {
        assert!(improves(1.0, 2.0, Better::Lower));
        assert!(!improves(2.0, 2.0, Better::Lower));
        assert!(improves(3.0, 2.0, Better::Higher));
    }
}
