//! Checking code before anyone relies on it - the reason Aegist can say
//! "I don't know" and mean it.
//!
//! A small model's words aren't evidence, so nothing it writes is trusted on
//! its say-so. Each piece of code goes through:
//!   syntax     the language's own compiler or parser accepts it
//!   names      every function, method and module it uses exists somewhere:
//!              in your project, in the code it learned from, or in the
//!              language itself. A name found nowhere was probably made up,
//!              and it says so, naming the name.
//!   tests      when there are tests to run, they pass
//!   confidence how likely the model found its own tokens, and how familiar
//!              the surrounding code was to it
//! and the verdict can only go down from there.

use crate::brain::Generation;
use crate::config::HonestySettings;
use crate::lang::{self, Lang};
use crate::proc::{self, Output};
use std::collections::{BTreeSet, HashSet};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Status {
    Pass,
    Fail(String),
    /// Couldn't check (no tool for it, or nothing to run).
    Skipped(String),
}

impl Status {
    pub fn passed(&self) -> bool {
        matches!(self, Status::Pass)
    }
    pub fn failed(&self) -> bool {
        matches!(self, Status::Fail(_))
    }
}

/// The first few lines of a tool's complaint, without the noise.
fn summarize(out: &Output, file: &Path, shown_as: &str) -> String {
    let text = out.combined().replace(&file.to_string_lossy().to_string(), shown_as);
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let keep: Vec<&str> = lines.iter().rev().take(6).rev().copied().collect();
    if keep.is_empty() {
        format!("exit code {}", out.code.map_or("?".into(), |c| c.to_string()))
    } else {
        keep.join("\n")
    }
}

/// Parse or compile `code` as `file_name` (a name only; it's checked in a
/// scratch folder) with the language's own tools.
pub fn syntax(lang: &Lang, code: &str, file_name: &str, timeout_s: u64) -> Status {
    match lang.id {
        "json" => {
            return match serde_json::from_str::<serde_json::Value>(code) {
                Ok(_) => Status::Pass,
                Err(e) => Status::Fail(e.to_string()),
            }
        }
        "yaml" => {
            return match serde_yaml::from_str::<serde_yaml::Value>(code) {
                Ok(_) => Status::Pass,
                Err(e) => Status::Fail(e.to_string()),
            }
        }
        _ => {}
    }
    let python_template: Vec<&str>;
    let template: Option<&[&str]> = if lang.id == "python" {
        match proc::python() {
            Some(py) => {
                python_template = std::iter::once(py).chain(lang.check[0][1..].iter().copied()).collect();
                Some(&python_template)
            }
            None => None,
        }
    } else {
        lang.check.iter().find(|t| proc::which(t[0]).is_some()).map(|t| &t[..])
    };
    let Some(template) = template else {
        return Status::Skipped(if lang.check.is_empty() {
            format!("no syntax checker for {}", lang.name)
        } else {
            format!("{} isn't installed", lang.check[0][0])
        });
    };
    let Ok(dir) = scratch_dir() else { return Status::Skipped("no scratch folder".into()) };
    let name = Path::new(file_name).file_name().and_then(|n| n.to_str()).unwrap_or("snippet");
    let file = dir.join(name);
    let out_file = dir.join("out.bin");
    if std::fs::write(&file, code).is_err() {
        return Status::Skipped("couldn't write a scratch file".into());
    }
    let mut cmd = std::process::Command::new(template[0]);
    for a in &template[1..] {
        cmd.arg(a.replace("{}", &file.to_string_lossy()).replace("{out}", &out_file.to_string_lossy()));
    }
    cmd.current_dir(&dir);
    let result = match proc::run(cmd, Some(Duration::from_secs(timeout_s)), None) {
        Ok(out) if out.ok() => Status::Pass,
        Ok(out) if out.timed_out => Status::Skipped(format!("{} took too long", template[0])),
        // Windows' "not recognized as a command"
        Ok(out) if out.code == Some(9009) => Status::Skipped(format!("{} isn't installed", template[0])),
        Ok(out) => Status::Fail(summarize(&out, &file, file_name)),
        Err(e) => Status::Skipped(e.to_string()),
    };
    let _ = std::fs::remove_dir_all(&dir);
    result
}

fn scratch_dir() -> std::io::Result<std::path::PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!("aegist-check-{}-{}", std::process::id(), N.fetch_add(1, Ordering::Relaxed)));
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

/// Names Aegist can vouch for: every name in the code it learned from and
/// every name in your project.
#[derive(Clone, Default)]
pub struct Names {
    pub learned: Arc<HashSet<String>>,
    pub project: Arc<HashSet<String>>,
}

impl Names {
    pub fn knows(&self, name: &str) -> bool {
        self.learned.contains(name) || self.project.contains(name)
    }
}

const DEFINERS: &[&str] = &[
    "def", "class", "fn", "function", "func", "let", "const", "var", "struct", "enum", "trait", "type", "interface", "impl", "mod",
    "module", "package", "as", "static", "local", "val", "namespace", "macro_rules",
];

/// Names `code` defines for itself: after a defining keyword, before an
/// assignment, or among a function's parameters and loop variables.
fn defined_in(code: &str) -> HashSet<&str> {
    let mut out = HashSet::new();
    let ids: Vec<(usize, &str)> = {
        let base = code.as_ptr() as usize;
        lang::identifiers(code).map(|w| (w.as_ptr() as usize - base, w)).collect()
    };
    let bytes = code.as_bytes();
    let next_non_space = |mut i: usize| {
        while i < bytes.len() && (bytes[i] == b' ' || bytes[i] == b'\t') {
            i += 1;
        }
        i
    };
    for (n, &(pos, word)) in ids.iter().enumerate() {
        if n > 0 && DEFINERS.contains(&ids[n - 1].1) {
            out.insert(word);
        }
        let after = next_non_space(pos + word.len());
        let assigned = bytes.get(after) == Some(&b'=') && bytes.get(after + 1) != Some(&b'=')
            || code[after..].starts_with(":=")
            || (bytes.get(after) == Some(&b':') && bytes.get(after + 1) != Some(&b':'));
        if assigned {
            out.insert(word);
        }
    }
    // parameters: the identifiers inside the parentheses right after a
    // definition, and loop / closure / exception variables
    let param_re = regex::Regex::new(r"(?:\bdef|\bfn|\bfunction|\bfunc)\s*\w*\s*(?:<[^>]*>)?\s*\(([^)]*)\)|\(([^()]*)\)\s*=>|\|([^|]*)\|\s*[{\w]|\bfor\s*\(?\s*(?:let|var|const)?\s*([\w\s,()]+?)\s+(?:in|of)\b|\bcatch\s*\(\s*(\w+)|\bexcept\b[^:]*\bas\s+(\w+)|\bwith\b[^:]*\bas\s+(\w+)").expect("valid regex");
    for cap in param_re.captures_iter(code) {
        for m in cap.iter().skip(1).flatten() {
            for piece in m.as_str().split(',') {
                if let Some(first) = lang::identifiers(piece).find(|w| !matches!(*w, "mut" | "self" | "const" | "let" | "var")) {
                    out.insert(first);
                }
            }
        }
    }
    out
}

/// Functions, methods and modules `code` uses that exist nowhere Aegist can
/// see: not in `code` itself, not in `context` (the rest of the file and
/// whatever it was shown), not in `names`, not built into the language.
/// Sorted, each once.
pub fn invented_names(code: &str, lang: Option<&Lang>, context: &str, names: &Names) -> Vec<String> {
    let defined = defined_in(code);
    let in_context: HashSet<&str> = lang::identifiers(context).collect();
    let used_re = regex::Regex::new(r"(?m)(?:\.\s*([A-Za-z_]\w*)\s*\(|\b([A-Za-z_]\w*)\s*\(|^\s*(?:import|from)\s+([A-Za-z_][\w.]*)|\brequire\(\s*['\x22]([\w./@-]+)['\x22]|\buse\s+([A-Za-z_]\w*)::)")
        .expect("valid regex");
    let strings_and_comments = strip_strings_and_comments(code, lang);
    let mut out = BTreeSet::new();
    for cap in used_re.captures_iter(&strings_and_comments) {
        let Some(m) = cap.iter().skip(1).flatten().next() else { continue };
        for word in m.as_str().split(['.', '/']).filter(|w| !w.is_empty() && !w.starts_with('@')) {
            let known = word.len() <= 1
                || defined.contains(word)
                || in_context.contains(word)
                || names.knows(word)
                || lang.is_some_and(|l| l.is_keyword(word))
                || lang::LANGS.iter().any(|l| l.keywords.contains(&word))
                || word.chars().all(|c| c.is_ascii_digit() || c == '_');
            if !known {
                out.insert(word.to_string());
            }
        }
    }
    out.into_iter().collect()
}

/// `code` with string literals and comments blanked out (same length), so
/// words inside them aren't mistaken for code.
pub fn strip_strings_and_comments(code: &str, lang: Option<&Lang>) -> String {
    let line = lang.and_then(|l| l.line_comment).unwrap_or("//");
    let block = lang.and_then(|l| l.block_comment);
    let chars: Vec<char> = code.chars().collect();
    let mut out = String::with_capacity(code.len());
    let mut i = 0;
    let starts = |i: usize, pat: &str| pat.chars().enumerate().all(|(k, c)| chars.get(i + k) == Some(&c));
    let blank = |c: char| if c == '\n' { '\n' } else { ' ' };
    while i < chars.len() {
        if starts(i, line) {
            while i < chars.len() && chars[i] != '\n' {
                out.push(' ');
                i += 1;
            }
        } else if let Some((a, b)) = block.filter(|(a, _)| starts(i, a)) {
            let close = (i + a.chars().count()..chars.len()).find(|&j| starts(j, b)).map_or(chars.len(), |j| j + b.chars().count());
            (i..close).for_each(|j| out.push(blank(chars[j])));
            i = close;
        } else if matches!(chars[i], '"' | '\'' | '`') {
            let q = chars[i];
            let triple = starts(i, &q.to_string().repeat(3));
            let (open_len, close_pat) = if triple { (3, q.to_string().repeat(3)) } else { (1, q.to_string()) };
            let mut j = i + open_len;
            while j < chars.len() && !starts(j, &close_pat) {
                if chars[j] == '\\' {
                    j += 1;
                } else if chars[j] == '\n' && !triple && q != '`' {
                    break;
                }
                j += 1;
            }
            let end = (j + if j < chars.len() && starts(j, &close_pat) { close_pat.len() } else { 0 }).min(chars.len());
            (i..end).for_each(|k| out.push(blank(chars[k])));
            i = end;
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    out
}

/// How much the model believed its own answer.
#[derive(Clone, Copy, Debug)]
pub struct Confidence {
    /// Average probability of the tokens it wrote (0-1).
    pub mean_prob: f32,
    /// Tokens below the "unsure" line.
    pub uncertain: usize,
    pub tokens: usize,
    /// The prompt's loss relative to the model's typical held-out loss:
    /// above ~1.3, the surrounding code is unlike what it learned from.
    pub unfamiliarity: Option<f32>,
}

impl Confidence {
    pub fn of(g: &Generation, typical_nll: Option<f32>, honesty: &HonestySettings) -> Self {
        Confidence {
            mean_prob: g.mean_prob(),
            uncertain: g.uncertain(honesty.uncertain_below),
            tokens: g.tokens(),
            unfamiliarity: typical_nll.filter(|t| *t > 0.0).map(|t| g.prompt_nll / t),
        }
    }

    /// 0-1, what the report shows: the tokens' mean probability, pulled
    /// down when the surrounding code was unfamiliar.
    pub fn score(&self) -> f32 {
        let penalty = self.unfamiliarity.map_or(1.0, |u| if u > 1.0 { (1.0 / u).powf(1.5) } else { 1.0 });
        (self.mean_prob * penalty).clamp(0.0, 1.0)
    }
}

/// What Aegist is willing to say about a piece of code.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Verdict {
    /// It fails a check, or confidence is too low: Aegist won't stand
    /// behind it and says it doesn't know.
    Refused,
    /// Passed what could be checked, but something couldn't be checked
    /// (no checker installed, no tests): offered, clearly marked unverified.
    Unverified,
    /// Syntax accepted and every name accounted for, with good confidence.
    Checked,
    /// All that, and tests ran and passed.
    Verified,
}

#[derive(Clone, Debug)]
pub struct Report {
    pub syntax: Status,
    pub invented: Vec<String>,
    pub tests: Status,
    pub confidence: Confidence,
    pub verdict: Verdict,
    /// Plain-language reasons, worst first.
    pub reasons: Vec<String>,
}

pub fn judge(syntax: Status, invented: Vec<String>, tests: Status, confidence: Confidence, honesty: &HonestySettings) -> Report {
    let mut reasons = Vec::new();
    let mut verdict = Verdict::Verified;
    let mut cap = |v: Verdict, why: String, reasons: &mut Vec<String>| {
        verdict = verdict.min(v);
        reasons.push(why);
    };
    match &syntax {
        Status::Fail(e) => cap(Verdict::Refused, format!("it doesn't pass the language's own check:\n{e}"), &mut reasons),
        Status::Skipped(why) if honesty.require_syntax_check => cap(Verdict::Unverified, format!("its syntax couldn't be checked ({why})"), &mut reasons),
        _ => {}
    }
    if invented.len() > honesty.max_invented_names {
        let list = invented.iter().map(|n| format!("`{n}`")).collect::<Vec<_>>().join(", ");
        cap(Verdict::Refused, format!("it uses {list}, which I can't find in your project, in anything I learned from, or in the language - \
                                       I probably made {} up", if invented.len() == 1 { "it" } else { "them" }), &mut reasons);
    }
    match &tests {
        Status::Fail(e) => cap(Verdict::Refused, format!("the tests fail:\n{e}"), &mut reasons),
        Status::Skipped(_) => cap(Verdict::Checked, String::new(), &mut Vec::new()),
        Status::Pass => {}
    }
    if confidence.tokens > 0 && confidence.score() < honesty.min_confidence {
        cap(Verdict::Refused, format!("I'm not confident in it ({:.0}% average certainty)", confidence.score() * 100.0), &mut reasons);
    } else if confidence.unfamiliarity.is_some_and(|u| u > 1.5) {
        cap(Verdict::Unverified, "the code around it is unlike anything I learned from".into(), &mut reasons);
    }
    if verdict == Verdict::Checked && !syntax.passed() {
        verdict = Verdict::Unverified;
    }
    Report { syntax, invented, tests, confidence, verdict, reasons }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(words: &[&str]) -> Names {
        Names { learned: Arc::new(words.iter().map(|w| w.to_string()).collect()), project: Arc::default() }
    }

    #[test]
    fn invented_names_are_caught_and_real_ones_are_not() {
        let py = lang::by_name("python");
        let code = "import json\nimport numpyy\n\ndef load(path):\n    data = json.loads(open(path).read())\n    total = helper(data)\n    \
                    return data.frobnicate_all(total)  # json.nothing()\n\ns = \"fake_call()\"\n";
        let found = invented_names(code, py, "def helper(x):\n    return x\n", &names(&["json", "loads"]));
        assert_eq!(found, vec!["frobnicate_all", "numpyy"]);
        let js = lang::by_name("javascript");
        let code = "const sum = (a, b) => a + b;\nfunction go(list) { return list.map(x => sum(x, 1)).quantumSort(); }\n";
        assert_eq!(invented_names(code, js, "", &names(&[])), vec!["quantumSort"]);
        // a definition that starts before the generated part still counts
        assert!(invented_names("def is_prime(n):\n    return n > 1 and all(n % d for d in range(2, n))\n", py, "", &names(&[])).is_empty());
        let rs = lang::by_name("rust");
        let code = "fn twice(v: &[i32]) -> Vec<i32> { v.iter().map(|x| x * 2).collect() }\n";
        assert!(invented_names(code, rs, "", &names(&[])).is_empty());
    }

    #[test]
    fn strings_and_comments_are_blanked() {
        let py = lang::by_name("python");
        let s = strip_strings_and_comments("x = 'a(b)' # call(1)\ny = \"\"\"doc\nmore()\"\"\"\nz()", py);
        assert!(!s.contains("call") && !s.contains("more") && s.contains("z()") && s.lines().count() == 4);
    }

    #[test]
    fn syntax_checks_use_the_language_tools() {
        let json = lang::by_name("json").unwrap();
        assert!(syntax(json, "{\"a\": [1, 2]}", "x.json", 5).passed());
        assert!(syntax(json, "{\"a\": [1, 2}", "x.json", 5).failed());
        if proc::python().is_some() {
            let py = lang::by_name("python").unwrap();
            assert!(syntax(py, "def f(x):\n    return x + 1\n", "m.py", 20).passed());
            match syntax(py, "def f(x)\n    return x +\n", "m.py", 20) {
                Status::Fail(e) => assert!(e.contains("m.py") && e.contains("SyntaxError"), "{e}"),
                other => panic!("{other:?}"),
            }
        }
        let asm = lang::by_name("asm").unwrap();
        assert!(matches!(syntax(asm, "mov eax, 1", "a.asm", 5), Status::Skipped(_)));
    }

    #[test]
    fn the_verdict_only_goes_down() {
        let h = HonestySettings::default();
        let sure = Confidence { mean_prob: 0.9, uncertain: 0, tokens: 20, unfamiliarity: Some(1.0) };
        assert_eq!(judge(Status::Pass, vec![], Status::Pass, sure, &h).verdict, Verdict::Verified);
        assert_eq!(judge(Status::Pass, vec![], Status::Skipped("no tests".into()), sure, &h).verdict, Verdict::Checked);
        assert_eq!(judge(Status::Skipped("no tool".into()), vec![], Status::Skipped("".into()), sure, &h).verdict, Verdict::Unverified);
        let r = judge(Status::Pass, vec!["magic".into()], Status::Pass, sure, &h);
        assert!(r.verdict == Verdict::Refused && r.reasons[0].contains("`magic`") && r.reasons[0].contains("made it up"));
        let unsure = Confidence { mean_prob: 0.2, ..sure };
        assert_eq!(judge(Status::Pass, vec![], Status::Pass, unsure, &h).verdict, Verdict::Refused);
        let strange = Confidence { unfamiliarity: Some(2.5), ..sure };
        assert!(strange.score() < sure.score());
        assert_eq!(judge(Status::Fail("bad".into()), vec![], Status::Pass, sure, &h).verdict, Verdict::Refused);
    }
}
