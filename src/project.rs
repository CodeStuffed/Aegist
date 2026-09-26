//! The project you're working in: its files, a search index over its code
//! (so the model is shown the definitions it needs from other files - its
//! memory reaches the whole repository, not just its context window), the
//! names it defines (so they're never mistaken for invented ones), and an
//! undo journal for every file Aegist writes.

use crate::lang;
use crate::learn;
use anyhow::{bail, Context, Result};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Files and bytes indexed at most (a huge repository is still usable;
/// only its first files are searched).
const MAX_FILES: usize = 20_000;
const MAX_BYTES: usize = 64 << 20;
const CHUNK_LINES: usize = 30;
const CHUNK_STEP: usize = 20;

#[derive(Clone, Debug)]
pub struct Snippet {
    pub path: String,
    /// 1-based, inclusive.
    pub first_line: usize,
    pub last_line: usize,
    pub text: String,
    pub score: f32,
}

struct Chunk {
    file: u32,
    first_line: u32,
    last_line: u32,
    len: u32,
}

#[derive(Default)]
struct Index {
    chunks: Vec<Chunk>,
    postings: HashMap<String, Vec<(u32, u16)>>,
    avg_len: f32,
}

/// Search terms in code: each identifier, lowercased, plus its parts
/// (snake_case and camelCase split), minus language keywords.
pub fn terms(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for id in lang::identifiers(text) {
        if id.len() < 2 || lang::LANGS.iter().any(|l| l.keywords.contains(&id)) {
            continue;
        }
        let lower = id.to_ascii_lowercase();
        let mut parts: Vec<String> = Vec::new();
        let mut cur = String::new();
        let chars: Vec<char> = id.chars().collect();
        for (i, &c) in chars.iter().enumerate() {
            let boundary = c == '_' || (c.is_ascii_uppercase() && i > 0 && (chars[i - 1].is_ascii_lowercase()
                || chars.get(i + 1).is_some_and(|n| n.is_ascii_lowercase()) && chars[i - 1].is_ascii_uppercase()));
            if boundary && !cur.is_empty() {
                parts.push(std::mem::take(&mut cur));
            }
            if c != '_' {
                cur.push(c.to_ascii_lowercase());
            }
        }
        if !cur.is_empty() {
            parts.push(cur);
        }
        if parts.len() > 1 {
            out.extend(parts.into_iter().filter(|p| p.len() >= 2 && *p != lower));
        }
        out.push(lower);
    }
    out
}

pub struct Project {
    pub root: PathBuf,
    /// Paths relative to root, with '/'.
    pub files: Vec<String>,
    contents: Vec<Arc<str>>,
    index: Index,
    pub names: Arc<HashSet<String>>,
    journal: Vec<Edit>,
}

#[derive(Clone, Debug)]
pub struct Edit {
    pub path: PathBuf,
    pub before: Option<String>,
    pub after: String,
    pub label: String,
}

/// The folder a project lives in: the nearest one up from `dir` holding
/// .git, else `dir` itself.
pub fn find_root(dir: &Path) -> PathBuf {
    dir.ancestors().find(|d| d.join(".git").exists()).unwrap_or(dir).to_path_buf()
}

impl Project {
    pub fn open(dir: &Path) -> Project {
        let root = find_root(dir);
        let mut p = Project { root: root.clone(), files: Vec::new(), contents: Vec::new(), index: Index::default(), names: Arc::default(),
                              journal: Vec::new() };
        let mut bytes = 0;
        let mut names = HashSet::new();
        // the home folder or a filesystem root isn't a project: don't index it all
        let too_broad = crate::config::home_dir().is_some_and(|h| h == root) || root.parent().is_none();
        if !too_broad {
            for f in learn::source_files(&root).into_iter().take(MAX_FILES) {
                let Ok(meta) = std::fs::metadata(&f) else { continue };
                if meta.len() > 1 << 20 || bytes + meta.len() as usize > MAX_BYTES {
                    continue;
                }
                let Ok(raw) = std::fs::read(&f) else { continue };
                if raw.contains(&0) {
                    continue;
                }
                let text = crate::corpus::clean_code(&raw);
                bytes += text.len();
                names.extend(lang::identifiers(&text).filter(|w| w.len() >= 2).map(str::to_string));
                let rel = f.strip_prefix(&root).unwrap_or(&f).to_string_lossy().replace('\\', "/");
                p.files.push(rel);
                p.contents.push(text.into());
            }
        }
        p.names = Arc::new(names);
        p.build_index();
        p
    }

    fn build_index(&mut self) {
        let mut idx = Index::default();
        let mut total_len = 0u64;
        for (fi, text) in self.contents.iter().enumerate() {
            let lines: Vec<&str> = text.lines().collect();
            let mut start = 0;
            while start < lines.len() {
                let end = (start + CHUNK_LINES).min(lines.len());
                let chunk_text = lines[start..end].join("\n");
                let mut tf: HashMap<String, u16> = HashMap::new();
                let words = terms(&chunk_text);
                for w in &words {
                    *tf.entry(w.clone()).or_default() += 1;
                }
                let id = idx.chunks.len() as u32;
                for (w, n) in tf {
                    idx.postings.entry(w).or_default().push((id, n));
                }
                idx.chunks.push(Chunk { file: fi as u32, first_line: start as u32 + 1, last_line: end as u32, len: words.len() as u32 });
                total_len += words.len() as u64;
                if end == lines.len() {
                    break;
                }
                start += CHUNK_STEP;
            }
        }
        idx.avg_len = total_len as f32 / idx.chunks.len().max(1) as f32;
        self.index = idx;
    }

    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// The code most relevant to `query` (BM25 over 30-line chunks), best
    /// first, at most one snippet per place, skipping file `exclude`.
    pub fn search(&self, query: &str, k: usize, exclude: Option<&str>) -> Vec<Snippet> {
        let idx = &self.index;
        let n = idx.chunks.len() as f32;
        let mut scores: HashMap<u32, f32> = HashMap::new();
        let mut qterms = terms(query);
        qterms.sort();
        qterms.dedup();
        for t in &qterms {
            let Some(post) = idx.postings.get(t) else { continue };
            let idf = ((n - post.len() as f32 + 0.5) / (post.len() as f32 + 0.5) + 1.0).ln();
            for &(c, tf) in post {
                let len = idx.chunks[c as usize].len as f32;
                let tf = tf as f32;
                *scores.entry(c).or_default() += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * len / idx.avg_len.max(1.0)));
            }
        }
        let mut ranked: Vec<(u32, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut out: Vec<Snippet> = Vec::new();
        for (c, score) in ranked {
            if out.len() >= k {
                break;
            }
            let ch = &idx.chunks[c as usize];
            let path = &self.files[ch.file as usize];
            if exclude == Some(path.as_str()) {
                continue;
            }
            // overlapping chunks of one file: keep the best
            if out.iter().any(|s| &s.path == path && (s.first_line as u32) <= ch.last_line && ch.first_line <= s.last_line as u32) {
                continue;
            }
            let text: String = self.contents[ch.file as usize]
                .lines()
                .skip(ch.first_line as usize - 1)
                .take((ch.last_line - ch.first_line + 1) as usize)
                .collect::<Vec<_>>()
                .join("\n");
            out.push(Snippet { path: path.clone(), first_line: ch.first_line as usize, last_line: ch.last_line as usize, text, score });
        }
        out
    }

    /// Relative path of a file (with '/'), or its full path if it's outside the project.
    pub fn rel(&self, path: &Path) -> String {
        path.strip_prefix(&self.root).unwrap_or(path).to_string_lossy().replace('\\', "/")
    }

    /// Write a file, remembering what was there so `undo` can put it back.
    pub fn write_file(&mut self, path: &Path, content: &str, label: &str) -> Result<()> {
        let before = std::fs::read_to_string(path).ok();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, content).with_context(|| format!("writing {}", path.display()))?;
        self.journal.push(Edit { path: path.to_path_buf(), before, after: content.to_string(), label: label.to_string() });
        self.refresh(path);
        Ok(())
    }

    /// Put back the file the last write changed. Refuses if the file has
    /// been changed since (by you or another program).
    pub fn undo(&mut self) -> Result<Option<Edit>> {
        let Some(edit) = self.journal.last().cloned() else { return Ok(None) };
        let now = std::fs::read_to_string(&edit.path).ok();
        if now.as_deref() != Some(edit.after.as_str()) {
            bail!("{} has changed since Aegist wrote it, so it wasn't undone (your newer changes would be lost).", edit.path.display());
        }
        match &edit.before {
            Some(text) => std::fs::write(&edit.path, text)?,
            None => std::fs::remove_file(&edit.path)?,
        }
        self.journal.pop();
        self.refresh(&edit.path);
        Ok(Some(edit))
    }

    pub fn history(&self) -> &[Edit] {
        &self.journal
    }

    /// Re-read one file into the index and names after it changed.
    pub fn refresh(&mut self, path: &Path) {
        let rel = self.rel(path);
        let text: Option<Arc<str>> = std::fs::read(path).ok().map(|b| crate::corpus::clean_code(&b).into());
        match (self.files.iter().position(|f| *f == rel), text) {
            (Some(i), Some(t)) => self.contents[i] = t,
            (Some(i), None) => {
                self.files.remove(i);
                self.contents.remove(i);
            }
            (None, Some(t)) if lang::is_code(path) => {
                self.files.push(rel);
                self.contents.push(t);
            }
            _ => return,
        }
        let mut names = (*self.names).clone();
        if let Some(i) = self.files.iter().position(|f| f == &self.rel(path)) {
            names.extend(lang::identifiers(&self.contents[i]).filter(|w| w.len() >= 2).map(str::to_string));
        }
        self.names = Arc::new(names);
        self.build_index();
    }

    /// A file's text, from the index if it's there.
    pub fn content(&self, rel: &str) -> Option<Arc<str>> {
        self.files.iter().position(|f| f == rel).map(|i| self.contents[i].clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn terms_split_identifiers() {
        let t = terms("parseHTTPResponse load_user_data x self return");
        for want in ["parsehttpresponse", "parse", "http", "response", "load_user_data", "load", "user", "data"] {
            assert!(t.contains(&want.to_string()), "{want} missing from {t:?}");
        }
        assert!(!t.contains(&"self".to_string()) && !t.contains(&"x".to_string()));
    }

    #[test]
    fn finds_relevant_code_and_undoes_writes() {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        std::fs::create_dir_all(root.join(".git")).unwrap();
        std::fs::create_dir_all(root.join("src")).unwrap();
        std::fs::write(root.join("src/users.py"), "def load_user(user_id):\n    return db.fetch_user(user_id)\n").unwrap();
        std::fs::write(root.join("src/math_utils.py"), "def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n").unwrap();
        std::fs::write(root.join("README.md"), "# demo\n").unwrap();
        let mut p = Project::open(&root.join("src"));
        assert_eq!(p.root, root);
        assert_eq!(p.file_count(), 3);
        let hits = p.search("user = load_user(42)", 5, None);
        assert_eq!(hits[0].path, "src/users.py");
        assert!(hits[0].text.contains("fetch_user"));
        assert!(p.search("user = load_user(42)", 5, Some("src/users.py")).iter().all(|h| h.path != "src/users.py"));
        assert!(p.names.contains("fetch_user") && p.names.contains("clamp"));

        let new = root.join("src/new.py");
        p.write_file(&new, "def added():\n    pass\n", "write").unwrap();
        assert!(p.names.contains("added") && p.search("added", 3, None)[0].path == "src/new.py");
        let users = root.join("src/users.py");
        p.write_file(&users, "changed\n", "edit").unwrap();
        assert_eq!(p.undo().unwrap().unwrap().label, "edit");
        assert!(std::fs::read_to_string(&users).unwrap().contains("load_user"));
        std::fs::write(&new, "edited by hand\n").unwrap();
        assert!(p.undo().unwrap_err().to_string().contains("changed since"));
        std::fs::write(&new, "def added():\n    pass\n").unwrap();
        p.undo().unwrap();
        assert!(!new.exists() && p.undo().unwrap().is_none());
    }
}
