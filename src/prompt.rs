//! Turning a request into what the model was trained on: files. A prompt
//! is the most relevant code from elsewhere in the project (each piece as
//! its own file), then the file being worked on - its path, and either the
//! code before and after a gap (fill-in-the-middle) or a comment saying
//! what the new file is for. Everything is measured in tokens and trimmed
//! to fit the context, nearest code kept first.

use crate::brain::Brain;
use crate::corpus::{self, EOT, FILE};
use crate::lang::Lang;
use crate::project::Project;

/// Room kept free in every prompt for safety.
const MARGIN: usize = 16;

fn floor_char(s: &str, mut i: usize) -> usize {
    i = i.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// The end of `text` that fits in `max_tokens`, starting at a line start.
pub fn keep_tail<'a>(brain: &Brain, text: &'a str, max_tokens: usize) -> &'a str {
    let ids = brain.tokenizer.encode(text);
    if ids.len() <= max_tokens {
        return text;
    }
    let kept: usize = ids[ids.len() - max_tokens..].iter().map(|&i| brain.tokenizer.token_len(i)).sum();
    let mut start = text.len().saturating_sub(kept);
    while !text.is_char_boundary(start) {
        start += 1;
    }
    match text[start..].find('\n') {
        Some(nl) if start + nl + 1 < text.len() => &text[start + nl + 1..],
        _ => &text[start..],
    }
}

/// The start of `text` that fits in `max_tokens`, ending at a line end.
pub fn keep_head<'a>(brain: &Brain, text: &'a str, max_tokens: usize) -> &'a str {
    let ids = brain.tokenizer.encode(text);
    if ids.len() <= max_tokens {
        return text;
    }
    let kept: usize = ids[..max_tokens].iter().map(|&i| brain.tokenizer.token_len(i)).sum();
    let end = floor_char(text, kept);
    match text[..end].rfind('\n') {
        Some(nl) if nl > 0 => &text[..nl + 1],
        _ => &text[..end],
    }
}

/// Code from elsewhere in the project relevant to `query`, as documents,
/// within `max_tokens`. Returns the text and the paths used.
pub fn related_code(brain: &Brain, project: Option<&Project>, query: &str, exclude: Option<&str>, max_tokens: usize) -> (String, Vec<String>) {
    let Some(project) = project else { return (String::new(), Vec::new()) };
    let mut out = String::new();
    let mut used = Vec::new();
    let mut left = max_tokens;
    for s in project.search(query, 8, exclude) {
        let doc = format!("{FILE}{}\n{}\n{EOT}", s.path, s.text);
        let n = brain.count_tokens(&doc);
        if n > left {
            continue;
        }
        left -= n;
        out.push_str(&doc);
        used.push(format!("{}:{}-{}", s.path, s.first_line, s.last_line));
    }
    (out, used)
}

pub struct Prompt {
    pub text: String,
    /// Where the related code came from ("path:first-last").
    pub sources: Vec<String>,
    /// The file's own code the model saw (before the gap, after it).
    pub before: String,
    pub after: String,
}

/// Fill the gap between `before` and `after` in file `path`.
pub fn infill(brain: &Brain, project: Option<&Project>, path: &str, before: &str, after: &str, max_new: usize) -> Prompt {
    let budget = brain.max_context().saturating_sub(max_new + MARGIN + brain.count_tokens(path) + 4);
    let local = budget * 7 / 10;
    let b = keep_tail(brain, before, local * 3 / 4);
    let a = keep_head(brain, after, local - brain.count_tokens(b).min(local * 3 / 4));
    let query: String = b.lines().rev().take(40).collect::<Vec<_>>().join("\n") + a.lines().take(10).collect::<Vec<_>>().join("\n").as_str();
    let used_local = brain.count_tokens(b) + brain.count_tokens(a);
    let (related, sources) = related_code(brain, project, &query, Some(path), budget.saturating_sub(used_local));
    Prompt { text: format!("{related}{}", corpus::fim_prompt(path, b, a)), sources, before: b.to_string(), after: a.to_string() }
}

/// How code for `description` in `lang` starts: a function request starts
/// with the language's function keyword, a class request with its class
/// keyword, a web page with the HTML skeleton. It keeps a small model on
/// the kind of code that was asked for.
pub fn starter(lang: Option<&Lang>, description: &str) -> &'static str {
    let d = description.to_lowercase();
    let wants = |w: &str| d.split(|c: char| !c.is_alphanumeric()).any(|x| x == w || x == format!("{w}s"));
    let Some(lang) = lang else { return "" };
    if lang.id == "html" {
        return "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n  <meta charset=\"utf-8\">\n";
    }
    if wants("class") {
        return match lang.id {
            "python" | "ruby" | "javascript" | "typescript" | "java" | "cs" | "cpp" | "kotlin" | "swift" | "scala" | "php" | "dart" => "class ",
            "rust" => "pub struct ",
            "go" => "type ",
            _ => "",
        };
    }
    if wants("function") || wants("method") || wants("func") || wants("def") {
        return match lang.id {
            "python" | "ruby" => "def ",
            "javascript" | "typescript" | "php" | "lua" => "function ",
            "rust" => "pub fn ",
            "go" | "swift" => "func ",
            "kotlin" => "fun ",
            "bash" => "function ",
            _ => "",
        };
    }
    ""
}

/// Start a new file at `path` whose first lines are a comment describing
/// it, then how that kind of code starts (`starter`).
pub fn new_file(brain: &Brain, project: Option<&Project>, path: &str, lang: Option<&Lang>, description: &str, max_new: usize) -> Prompt {
    let comment = match lang {
        Some(l) => l.comment(description.trim()),
        None => description.trim().to_string(),
    };
    let start = starter(lang, description);
    let header = if lang.is_some_and(|l| l.id == "html") {
        format!("{start}{comment}\n")
    } else if start.is_empty() {
        format!("{comment}\n")
    } else {
        format!("{comment}\n{start}")
    };
    let budget = brain.max_context().saturating_sub(max_new + MARGIN + brain.count_tokens(&header) + brain.count_tokens(path) + 4);
    let (related, sources) = related_code(brain, project, description, Some(path), budget);
    Prompt { text: format!("{related}{FILE}{path}\n{header}"), sources, before: header, after: String::new() }
}

/// Continue file `path` after `before` (to the end of the file).
pub fn continuation(brain: &Brain, project: Option<&Project>, path: &str, before: &str, max_new: usize) -> Prompt {
    let budget = brain.max_context().saturating_sub(max_new + MARGIN + brain.count_tokens(path) + 4);
    let b = keep_tail(brain, before, budget * 7 / 10);
    let query: String = b.lines().rev().take(40).collect::<Vec<_>>().join("\n");
    let (related, sources) = related_code(brain, project, &query, Some(path), budget.saturating_sub(brain.count_tokens(b)));
    Prompt { text: format!("{related}{FILE}{path}\n{b}"), sources, before: b.to_string(), after: String::new() }
}

/// Stop a gap's code where it starts repeating the code that follows the
/// gap (a model filling a gap sometimes runs on into what's already there).
pub fn stop_before(after: &str) -> impl Fn(&str) -> Option<usize> + Sync + '_ {
    let anchor = after.lines().map(str::trim).find(|l| l.len() >= 4).map(str::to_string);
    move |text: &str| {
        let anchor = anchor.as_deref()?;
        let mut offset = 0;
        for line in text.split_inclusive('\n') {
            if line.trim() == anchor && line.ends_with('\n') {
                return Some(offset);
            }
            offset += line.len();
        }
        None
    }
}

/// A file path for new code described as `description` in `lang`, like
/// "prime_numbers.py" for "check whether a number is prime" in Python.
pub fn suggest_path(description: &str, lang: &Lang) -> String {
    const SKIP: &[&str] = &[
        "a", "an", "the", "that", "which", "write", "create", "make", "build", "generate", "code", "program", "script", "function", "in",
        "for", "to", "of", "and", "with", "me", "please", "some", "simple", "small", "new", "file", "using", "is", "it", "my", "can",
        "whether", "if", "on", "into", "from", "by", "this", "at", "be", "should", "will", "would", "i", "want", "need", "app",
    ];
    let words: Vec<String> = description
        .split(|c: char| !c.is_alphanumeric())
        .map(|w| w.to_lowercase())
        .filter(|w| !w.is_empty() && !SKIP.contains(&w.as_str()) && crate::lang::by_name(w).is_none_or(|l| l.name != lang.name))
        .take(3)
        .collect();
    let stem = if words.is_empty() { "main".to_string() } else { words.join("_") };
    let stem = match lang.id {
        "javascript" | "typescript" | "html" | "css" => stem.replace('_', "-"),
        _ => stem,
    };
    let stem = if lang.id == "html" && words.is_empty() { "index".into() } else { stem };
    format!("{stem}.{}", lang.exts.first().copied().unwrap_or("txt"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::by_name;

    #[test]
    fn paths_follow_the_description_and_language() {
        let py = by_name("python").unwrap();
        assert_eq!(suggest_path("write a function that checks whether a number is prime", py), "checks_number_prime.py");
        assert_eq!(suggest_path("snake game", by_name("js").unwrap()), "snake-game.js");
        assert_eq!(suggest_path("write python code", py), "main.py");
        assert_eq!(suggest_path("a website", by_name("html").unwrap()), "index.html");
    }

    #[test]
    fn requests_start_the_kind_of_code_asked_for() {
        assert_eq!(starter(by_name("python"), "a function that adds numbers"), "def ");
        assert_eq!(starter(by_name("rust"), "write functions to parse dates"), "pub fn ");
        assert_eq!(starter(by_name("python"), "a Stack class"), "class ");
        assert_eq!(starter(by_name("python"), "a script that prints hello"), "");
        assert!(starter(by_name("html"), "a snake game").starts_with("<!DOCTYPE html>"));
    }

    #[test]
    fn repeating_what_follows_the_gap_stops_it() {
        let rule = stop_before("\n    return total\n}\n");
        assert_eq!(rule("    total += x\n    return total\n"), Some(15));
        assert_eq!(rule("    total += x\n"), None);
        assert_eq!(stop_before("")("anything\n"), None);
    }

    #[test]
    fn prompts_fit_and_keep_the_nearest_code() {
        let (_tmp, s) = crate::trainer::tests::trained(5);
        let brain = Brain::load(&s).unwrap();
        let before = (0..400).map(|i| format!("x{i} = {i}\n")).collect::<String>();
        let after = (0..400).map(|i| format!("y{i} = {i}\n")).collect::<String>();
        let p = infill(&brain, None, "big.py", &before, &after, 32);
        assert!(brain.count_tokens(&p.text) + 32 <= brain.max_context());
        assert!(p.before.ends_with("x399 = 399\n") && p.before.starts_with('x'));
        assert!(p.after.starts_with("y0 = 0\n") && p.after.ends_with('\n'));
        assert!(p.text.contains(&format!("{FILE}big.py\n")));
    }
}
