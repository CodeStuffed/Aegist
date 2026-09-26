//! What Aegist does with a request: write a new file, complete code in the
//! middle of one, or fix a failing command. Each writes several candidates
//! (one careful, the rest more adventurous), checks every one of them, and
//! offers only the best that passes - or says plainly that none did.

use crate::brain::{Brain, GenOptions, Generation};
use crate::config::Settings;
use crate::lang::{self, Lang};
use crate::proc;
use crate::project::Project;
use crate::prompt::{self, Prompt};
use crate::verify::{self, Confidence, Names, Report, Status, Verdict};
use anyhow::{bail, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

/// Progress, for showing the work as it happens.
#[derive(Clone, Debug)]
pub enum Progress {
    /// Reading the prompt: its size in tokens, and where related code came from.
    Reading { tokens: usize, sources: Vec<String> },
    /// New text from the first (most careful) candidate, and its probability.
    Text(String, f32),
    /// Checking candidate i of n.
    Checking { candidate: usize, of: usize, what: &'static str },
    /// Running a command (fix): the command, attempt number.
    Running { command: String, attempt: usize },
    Note(String),
}

pub type OnProgress<'a> = &'a (dyn Fn(Progress) + Sync);

#[derive(Clone, Debug)]
pub struct Candidate {
    /// The code the model wrote.
    pub code: String,
    /// The whole file with it in place.
    pub file: String,
    pub generation: Generation,
    pub report: Report,
}

#[derive(Clone, Debug)]
pub struct Answer {
    pub path: PathBuf,
    /// Path shown to the user.
    pub shown: String,
    /// The file before (None: a new file).
    pub original: Option<String>,
    /// Best first.
    pub candidates: Vec<Candidate>,
    pub sources: Vec<String>,
    pub prompt_tokens: usize,
}

impl Answer {
    pub fn best(&self) -> Option<&Candidate> {
        self.candidates.first()
    }

    /// The best candidate, if Aegist is willing to stand behind it.
    pub fn accepted(&self) -> Option<&Candidate> {
        self.best().filter(|c| c.report.verdict > Verdict::Refused)
    }
}

pub struct Assistant<'a> {
    pub brain: &'a Brain,
    pub settings: &'a Settings,
    pub names: Names,
    pub cancel: &'a AtomicBool,
}

impl Assistant<'_> {
    /// One careful candidate (temperature from settings) plus adventurous ones.
    fn options(&self, max_new: usize, seed: u64) -> Vec<GenOptions> {
        let base = GenOptions { max_new, seed, ..GenOptions::from_settings(self.settings) };
        let n = self.settings.honesty.samples.max(1);
        (0..n)
            .map(|i| if i == 0 { base.clone() } else { GenOptions { temperature: base.temperature.max(0.7), seed: seed + i as u64, ..base.clone() } })
            .collect()
    }

    fn cancelled(&self) -> bool {
        self.cancel.load(Ordering::Relaxed)
    }

    /// Write every candidate, check each, best first.
    #[allow(clippy::too_many_arguments)]
    fn candidates(&self, prompt: &Prompt, lang: Option<&'static Lang>, shown: &str, max_new: usize, assemble: &dyn Fn(&str) -> String,
                  stop: &(dyn Fn(&str) -> Option<usize> + Sync), tests: Option<&dyn Fn(&str) -> Status>, progress: OnProgress) -> (Vec<Candidate>, usize) {
        let tokens = self.brain.count_tokens(&prompt.text);
        progress(Progress::Reading { tokens, sources: prompt.sources.clone() });
        let seed = crate::rng::Rng::from_time().next_u64();
        let cancel = self.cancel;
        let gens = self.brain.generate_many_streaming(&prompt.text, &self.options(max_new, seed), stop, &|i, text, p| {
            if i == 0 {
                progress(Progress::Text(text.to_string(), p));
            }
            !cancel.load(Ordering::Relaxed)
        });
        let context = format!("{}{}{}", prompt.text, prompt.before, prompt.after);
        let n = gens.len();
        let mut seen = std::collections::HashSet::new();
        let mut out: Vec<Candidate> = Vec::new();
        for (i, g) in gens.into_iter().enumerate() {
            if self.cancelled() {
                break;
            }
            if g.text.trim().is_empty() || !seen.insert(g.text.clone()) {
                continue; // nothing written, or an identical candidate was already checked
            }
            let file = assemble(&g.text);
            progress(Progress::Checking { candidate: i + 1, of: n, what: "syntax" });
            let syntax = match lang {
                Some(l) => verify::syntax(l, &file, shown, self.settings.honesty.check_timeout_s),
                None => Status::Skipped("unknown language".into()),
            };
            progress(Progress::Checking { candidate: i + 1, of: n, what: "names" });
            let invented = verify::invented_names(&g.text, lang, &context, &self.names);
            let confidence = Confidence::of(&g, self.brain.typical_nll(), &self.settings.honesty);
            let tests_status = match tests {
                Some(run) if syntax.passed() && invented.is_empty() => {
                    progress(Progress::Checking { candidate: i + 1, of: n, what: "tests" });
                    run(&file)
                }
                Some(_) => Status::Skipped("not run: the code failed an earlier check".into()),
                None => Status::Skipped("no tests".into()),
            };
            let report = verify::judge(syntax, invented, tests_status, confidence, &self.settings.honesty);
            out.push(Candidate { code: g.text.clone(), file, generation: g, report });
        }
        out.sort_by(|a, b| {
            b.report.verdict.cmp(&a.report.verdict)
                .then(a.report.invented.len().cmp(&b.report.invented.len()))
                .then(b.generation.mean_logprob().total_cmp(&a.generation.mean_logprob()))
        });
        (out, tokens)
    }

    /// A new file at `path` doing what `description` says.
    pub fn write_new(&self, project: Option<&Project>, path: &Path, shown: &str, description: &str, progress: OnProgress) -> Result<Answer> {
        let lang = lang::for_path(path);
        let max_new = self.settings.inference.max_new_tokens;
        let prompt = prompt::new_file(self.brain, project, shown, lang, description, max_new);
        let header = prompt.before.clone();
        let (candidates, prompt_tokens) =
            self.candidates(&prompt, lang, shown, max_new, &|code| format!("{header}{code}"), &crate::brain::never_stop, None, progress);
        Ok(Answer { path: path.to_path_buf(), shown: shown.to_string(), original: std::fs::read_to_string(path).ok(), candidates,
                    sources: prompt.sources, prompt_tokens })
    }

    /// Fill in code at byte `at` of file `path` (whose text is `text`).
    pub fn complete(&self, project: Option<&Project>, path: &Path, shown: &str, text: &str, at: usize, progress: OnProgress) -> Result<Answer> {
        if at > text.len() || !text.is_char_boundary(at) {
            bail!("that position isn't inside {shown}");
        }
        let lang = lang::for_path(path);
        let max_new = self.settings.inference.max_new_tokens;
        let (before, after) = text.split_at(at);
        let prompt = if after.trim().is_empty() {
            prompt::continuation(self.brain, project, shown, before, max_new)
        } else {
            prompt::infill(self.brain, project, shown, before, after, max_new)
        };
        let stop = prompt::stop_before(after);
        let (candidates, prompt_tokens) =
            self.candidates(&prompt, lang, shown, max_new, &|code| format!("{before}{code}{after}"), &stop, None, progress);
        Ok(Answer { path: path.to_path_buf(), shown: shown.to_string(), original: Some(text.to_string()), candidates, sources: prompt.sources,
                    prompt_tokens })
    }

    /// Make `command` pass by rewriting the code around where it fails.
    /// Every attempt is written, tested, and put back unless it passes.
    pub fn fix(&self, project: &mut Project, command: &str, progress: OnProgress) -> Result<FixOutcome> {
        let timeout = Duration::from_secs(300);
        let run = |attempt: usize| -> Result<proc::Output> {
            progress(Progress::Running { command: command.to_string(), attempt });
            let mut cmd = proc::shell(command);
            cmd.current_dir(&project.root);
            proc::run(cmd, Some(timeout), Some(self.cancel))
        };
        let first = run(0)?;
        if first.ok() {
            return Ok(FixOutcome::AlreadyPassing(first));
        }
        let places = failure_locations(&first.combined(), &project.root);
        if places.is_empty() {
            return Ok(FixOutcome::NoLocation(first));
        }
        let mut tried = 0;
        let max_new = 160;
        for (path, line) in places.iter().take(3) {
            let Ok(original) = std::fs::read_to_string(path) else { continue };
            let shown = project.rel(path);
            let lang = lang::for_path(path);
            for span in spans_around(&original, *line) {
                if self.cancelled() {
                    return Ok(FixOutcome::Cancelled { tried });
                }
                let (before, old, after) = (&original[..span.0], &original[span.0..span.1], &original[span.1..]);
                progress(Progress::Note(format!("rewriting {shown} lines {}-{}", line_of(&original, span.0), line_of(&original, span.1.max(1) - 1))));
                let prompt = prompt::infill(self.brain, Some(project), &shown, before, after, max_new);
                let stop = prompt::stop_before(after);
                let assemble = |code: &str| format!("{before}{code}{after}");
                let (cands, _) = self.candidates(&prompt, lang, &shown, max_new, &assemble, &stop, None, progress);
                for c in cands.iter().filter(|c| c.code != old && !c.report.syntax.failed() && c.report.invented.is_empty()) {
                    tried += 1;
                    std::fs::write(path, &c.file)?;
                    let result = run(tried);
                    if result.as_ref().is_ok_and(|r| r.ok()) {
                        std::fs::write(path, &original)?;
                        project.write_file(path, &c.file, &format!("fix: {command}"))?;
                        return Ok(FixOutcome::Fixed { path: path.clone(), shown, before: original.clone(), candidate: c.clone(), output: result? });
                    }
                    std::fs::write(path, &original)?;
                    if self.cancelled() {
                        return Ok(FixOutcome::Cancelled { tried });
                    }
                }
            }
        }
        Ok(FixOutcome::NotFixed { tried, output: first, places })
    }
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)] // made once per /fix
pub enum FixOutcome {
    AlreadyPassing(proc::Output),
    /// The failure didn't point at a file in the project.
    NoLocation(proc::Output),
    Fixed { path: PathBuf, shown: String, before: String, candidate: Candidate, output: proc::Output },
    NotFixed { tried: usize, output: proc::Output, places: Vec<(PathBuf, usize)> },
    Cancelled { tried: usize },
}

fn line_of(text: &str, byte: usize) -> usize {
    text[..byte.min(text.len())].matches('\n').count() + 1
}

/// Byte ranges to rewrite around 1-based `line`: the line itself, the lines
/// around it, then its whole block (by indentation).
fn spans_around(text: &str, line: usize) -> Vec<(usize, usize)> {
    let starts: Vec<usize> = std::iter::once(0).chain(text.match_indices('\n').map(|(i, _)| i + 1)).filter(|&i| i <= text.len()).collect();
    let nlines = starts.len();
    if line == 0 || line > nlines {
        return Vec::new();
    }
    let range = |a: usize, b: usize| -> (usize, usize) {
        let a = a.clamp(1, nlines);
        let b = b.clamp(a, nlines);
        (starts[a - 1], if b < nlines { starts[b] } else { text.len() })
    };
    let lines: Vec<&str> = text.lines().collect();
    let indent = |l: usize| lines.get(l - 1).map(|s| s.len() - s.trim_start().len());
    let mut out = vec![range(line, line), range(line.saturating_sub(2), line + 2)];
    // the block: the lines around at least as indented as this one
    if let Some(ind) = indent(line).filter(|&i| i > 0) {
        let mut a = line;
        while a > 1 && indent(a - 1).is_some_and(|i| i >= ind || lines[a - 2].trim().is_empty()) && line - a < 15 {
            a -= 1;
        }
        let mut b = line;
        while b < lines.len() && indent(b + 1).is_some_and(|i| i >= ind || lines[b].trim().is_empty()) && b - line < 15 {
            b += 1;
        }
        out.push(range(a, b));
    }
    out.dedup();
    out
}

/// Files and lines an error message points at, inside `root`, most
/// relevant first (a traceback's deepest frame in your code; a compiler's
/// first error).
pub fn failure_locations(output: &str, root: &Path) -> Vec<(PathBuf, usize)> {
    let patterns = [
        // Python traceback: File "x.py", line 12
        (r#"File "([^"]+)", line (\d+)"#, true),
        // Rust: --> src/main.rs:12:5
        (r"-->\s+([^\s:]+):(\d+):\d+", false),
        // gcc/clang/go/tsc/eslint and friends: path:12:5
        (r"(?m)^\s*([^\s:()]+\.[A-Za-z0-9]+):(\d+)(?::\d+)?", false),
        // JavaScript stack: at fn (path:12:5)
        (r"\(([^()\s]+\.[A-Za-z0-9]+):(\d+):\d+\)", false),
        // pytest short: path:12: in test_x
        (r"(?m)^([^\s:]+\.py):(\d+):", false),
    ];
    let mut found: Vec<(PathBuf, usize)> = Vec::new();
    for (pat, deepest_first) in patterns {
        let re = regex::Regex::new(pat).expect("valid regex");
        let mut here: Vec<(PathBuf, usize)> = Vec::new();
        for cap in re.captures_iter(output) {
            let raw = cap[1].trim_start_matches("file://");
            let Ok(line) = cap[2].parse::<usize>() else { continue };
            let p = Path::new(raw);
            let full = if p.is_absolute() { p.to_path_buf() } else { root.join(p) };
            let Ok(full) = full.canonicalize() else { continue };
            let Ok(root_c) = root.canonicalize() else { continue };
            if full.starts_with(&root_c) && full.is_file() && !here.contains(&(full.clone(), line)) {
                here.push((full, line));
            }
        }
        if deepest_first {
            here.reverse();
        }
        for h in here {
            if !found.contains(&h) {
                found.push(h);
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_messages_point_at_project_files() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("app.py"), "x\n").unwrap();
        std::fs::write(root.join("lib.py"), "y\n").unwrap();
        std::fs::write(root.join("src/main.rs"), "z\n").unwrap();
        let py = format!("Traceback (most recent call last):\n  File \"{}/app.py\", line 3, in <module>\n  File \"{}/lib.py\", line 7, in f\n  \
                          File \"/usr/lib/python3/json.py\", line 9\nValueError", root.display(), root.display());
        let found = failure_locations(&py, root);
        assert_eq!(found.iter().map(|(p, l)| (p.file_name().unwrap().to_str().unwrap(), *l)).collect::<Vec<_>>(), [("lib.py", 7), ("app.py", 3)]);
        let rs = "error[E0308]: mismatched types\n  --> src/main.rs:12:5\n";
        assert_eq!(failure_locations(rs, root)[0].1, 12);
        assert!(failure_locations("all good", root).is_empty());
    }

    #[test]
    fn spans_grow_from_the_line_to_its_block() {
        let text = "def f(x):\n    a = 1\n    b = 2\n    return a + b\n\nprint(f(1))\n";
        let spans = spans_around(text, 3);
        assert_eq!(&text[spans[0].0..spans[0].1], "    b = 2\n");
        assert_eq!(&text[spans[1].0..spans[1].1], "def f(x):\n    a = 1\n    b = 2\n    return a + b\n\n");
        let block = spans.last().unwrap();
        assert!(text[block.0..block.1].starts_with("    a = 1") && text[block.0..block.1].contains("return"));
        assert!(spans_around(text, 99).is_empty());
    }
}
