//! The training corpus: source files `aegist learn` collected under
//! data/corpus/<source>/, and their tokens, cached per file so a growing
//! corpus stays cheap to reload.
//!
//! Each file becomes one document: a file marker, its path (the model
//! learns which language and project it's in from that), the code, and an
//! end marker. Some files instead become fill-in-the-middle examples - the
//! code before a gap, the code after it, then the gap itself - which is how
//! the model learns to complete code in the middle of a file.
//!
//! All the tokens are joined into one file on disk (brain/token_cache/all-*.bin)
//! that training memory-maps instead of loading: billions of tokens cost
//! disk space, not RAM, and the operating system keeps the hot parts cached.

use crate::config::Settings;
use crate::rng::Rng;
use crate::tokenizer::Tokenizer;
use crate::util::{fnv1a, write_atomic};
use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// Start of a file: followed by its path and a newline.
pub const FILE: char = '\u{1c}';
/// Fill-in-the-middle: the code before the gap follows.
pub const PRE: char = '\u{1}';
/// Fill-in-the-middle: the code after the gap follows.
pub const SUF: char = '\u{2}';
/// Fill-in-the-middle: the gap's code follows.
pub const MID: char = '\u{3}';
/// End of a file (or of a gap's code).
pub const EOT: char = '\u{4}';

pub fn corpus_dir(settings: &Settings) -> PathBuf {
    settings.data_path("corpus")
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.is_file() && crate::lang::is_code(&p) {
            out.push(p);
        }
    }
}

pub fn corpus_files(settings: &Settings) -> Vec<PathBuf> {
    let mut files = Vec::new();
    walk(&corpus_dir(settings), &mut files);
    files.sort();
    files
}

/// Code as the model sees it: UTF-8, `\n` line ends, no marker characters.
pub fn clean_code(bytes: &[u8]) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\r' {
            if chars.peek() != Some(&'\n') {
                out.push('\n');
            }
        } else if c.is_control() && !c.is_whitespace() {
            continue;
        } else {
            out.push(c);
        }
    }
    out
}

fn read(path: &Path) -> String {
    clean_code(&std::fs::read(path).unwrap_or_default())
}

/// A whole file as a training document.
pub fn document(path: &str, code: &str) -> String {
    format!("{FILE}{path}\n{code}{EOT}")
}

/// A fill-in-the-middle document: `prefix` + gap + `suffix` is the file.
pub fn fim_document(path: &str, prefix: &str, suffix: &str, middle: &str) -> String {
    format!("{FILE}{path}\n{PRE}{prefix}{SUF}{suffix}{MID}{middle}{EOT}")
}

/// The prompt that asks for the gap between `prefix` and `suffix`: the model
/// continues it with the gap's code, then EOT.
pub fn fim_prompt(path: &str, prefix: &str, suffix: &str) -> String {
    format!("{FILE}{path}\n{PRE}{prefix}{SUF}{suffix}{MID}")
}

/// How files become fill-in-the-middle examples.
#[derive(Clone, Copy, Debug)]
pub struct Fim {
    /// Share of files trained this way.
    pub rate: f64,
    /// Longest piece of a file per example, in characters (about what fits
    /// in the model's context); longer files are split into pieces.
    pub window_chars: usize,
}

fn floor_char(s: &str, mut i: usize) -> usize {
    i = i.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Split at line ends into pieces of at most `max` bytes (a single longer
/// line becomes its own piece).
fn pieces(code: &str, max: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < code.len() {
        if code.len() - start <= max {
            out.push(&code[start..]);
            break;
        }
        let limit = floor_char(code, start + max);
        let end = code[start..limit].rfind('\n').map(|i| start + i + 1).filter(|&e| e > start).unwrap_or_else(|| {
            code[limit..].find('\n').map(|i| limit + i + 1).unwrap_or(code.len())
        });
        out.push(&code[start..end]);
        start = end;
    }
    out
}

/// A file's training documents. Deterministic in the path, so re-encoding
/// an unchanged file gives the same tokens.
pub fn file_documents(path: &str, code: &str, fim: Fim) -> Vec<String> {
    let mut rng = Rng::new(fnv1a(path.as_bytes()) ^ 0xf1f1);
    if code.trim().is_empty() {
        return Vec::new();
    }
    if rng.uniform() as f64 >= fim.rate {
        return vec![document(path, code)];
    }
    pieces(code, fim.window_chars.max(64))
        .into_iter()
        .map(|piece| {
            let len = piece.len();
            let (mut a, mut b) = (rng.below(len + 1), rng.below(len + 1));
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            if rng.below(2) == 0 {
                // gap on whole lines: what completing a line or a block looks like
                a = piece[..a].rfind('\n').map_or(0, |i| i + 1);
                b = piece[b..].find('\n').map_or(len, |i| b + i + 1);
            }
            let (a, b) = (floor_char(piece, a), floor_char(piece, b));
            fim_document(path, &piece[..a], &piece[b..], &piece[a..b])
        })
        .collect()
}

/// A corpus file's path as the model sees it: without the source folder
/// (data/corpus/<source>/src/x.py is "src/x.py").
pub fn display_path(rel: &str) -> &str {
    rel.split_once('/').map_or(rel, |(_, rest)| rest)
}

/// Up to `max_chars` of the corpus as one string, to learn the vocabulary
/// from. If the corpus is bigger, every file gives the same share of its
/// size, so the sample looks like the whole corpus, not just its first files.
pub fn read_corpus(settings: &Settings, max_chars: usize) -> String {
    let files = corpus_files(settings);
    let sizes: Vec<u64> = files.iter().map(|f| std::fs::metadata(f).map(|m| m.len()).unwrap_or(0)).collect();
    let total: u64 = sizes.iter().sum();
    let mut out = String::new();
    for (f, &size) in files.iter().zip(&sizes) {
        let text = if total as usize <= max_chars {
            read(f)
        } else {
            let share = (size as u128 * max_chars as u128 / total.max(1) as u128) as u64;
            let mut bytes = Vec::new();
            if share == 0 || File::open(f).and_then(|h| h.take(share).read_to_end(&mut bytes)).is_err() {
                continue;
            }
            if (share as usize) < size as usize {
                // end on a whole line
                let cut = bytes.iter().rposition(|&b| b == b'\n').unwrap_or(0);
                bytes.truncate(cut + 1);
            }
            clean_code(&bytes)
        };
        if text.trim().is_empty() {
            continue;
        }
        out.push_str(&text);
        if !out.ends_with('\n') {
            out.push('\n');
        }
    }
    let end = floor_char(&out, max_chars);
    out.truncate(end);
    out
}

#[derive(Default, Serialize, Deserialize)]
struct Manifest {
    /// merges + document format: tokens cached under another are stale
    format_hash: u64,
    /// file -> (size, modified) when its tokens were cached
    files: BTreeMap<String, (u64, u128)>,
    /// the files whose tokens are in the joined file, in order, with their token counts
    #[serde(default)]
    order: Vec<(String, (u64, u128), u64)>,
    /// the joined file's name; a rebuild writes a new one, because Windows
    /// won't replace or delete a file that is memory-mapped
    #[serde(default)]
    joined: String,
}

/// The corpus's tokens, memory-mapped from disk; use it like a `&[u16]`.
pub struct Tokens {
    map: Option<memmap2::Mmap>,
}

#[cfg(target_endian = "big")]
compile_error!("token files are little-endian");

impl std::ops::Deref for Tokens {
    type Target = [u16];
    fn deref(&self) -> &[u16] {
        match &self.map {
            // Safety: mappings start page-aligned, and the file holds u16s
            // written little-endian (checked above) and is never truncated
            // (rebuilds write a new file; old ones are only deleted).
            Some(m) => unsafe { std::slice::from_raw_parts(m.as_ptr() as *const u16, m.len() / 2) },
            None => &[],
        }
    }
}

fn u16_bytes(ids: &[u16]) -> Vec<u8> {
    ids.iter().flat_map(|i| i.to_le_bytes()).collect()
}

fn stamp(p: &Path) -> (u64, u128) {
    let m = std::fs::metadata(p).ok();
    let size = m.as_ref().map(|m| m.len()).unwrap_or(0);
    let mtime = m
        .and_then(|m| m.modified().ok())
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    (size, mtime)
}

/// Token ids for the whole corpus, re-encoding only files that changed.
pub fn corpus_tokens(tok: &Tokenizer, settings: &Settings, fim: Fim) -> Result<Tokens> {
    assert!(tok.vocab_size() <= u16::MAX as usize + 1);
    let root = corpus_dir(settings);
    let cache = settings.data_path("brain/token_cache");
    std::fs::create_dir_all(&cache)?;
    // one process at a time (e.g. `learn` and `train` in two terminals)
    let lock = File::create(cache.join(".lock"))?;
    lock.lock()?;
    let manifest_path = cache.join("manifest.json");
    let format = format!("code-docs-1 {} {}", fim.rate, fim.window_chars);
    let format_hash = fnv1a(serde_json::to_string(&tok.merges)?.as_bytes()) ^ fnv1a(format.as_bytes());
    let manifest: Manifest = std::fs::read(&manifest_path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .filter(|m: &Manifest| m.format_hash == format_hash)
        .unwrap_or(Manifest { format_hash, ..Default::default() });

    let files = corpus_files(settings);
    let entries: Vec<(String, PathBuf, (u64, u128), PathBuf)> = files
        .iter()
        .map(|f| {
            let rel = f.strip_prefix(&root).unwrap_or(f).to_string_lossy().replace('\\', "/");
            let bin = cache.join(format!("{:016x}.bin", fnv1a(rel.as_bytes())));
            (rel, f.clone(), stamp(f), bin)
        })
        .collect();
    // every file's own token cache, brought up to date in parallel
    let counts: Vec<Result<u64>> = entries
        .par_iter()
        .map(|(rel, path, st, bin)| {
            if manifest.files.get(rel) == Some(st) {
                if let Ok(m) = std::fs::metadata(bin) {
                    return Ok(m.len() / 2);
                }
            }
            let docs = file_documents(display_path(rel), &read(path), fim);
            let ids: Vec<u16> = docs.iter().flat_map(|d| tok.encode(d)).map(|i| i as u16).collect();
            write_atomic(bin, &u16_bytes(&ids))?;
            Ok(ids.len() as u64)
        })
        .collect();
    let mut current: HashMap<&str, ((u64, u128), u64, &Path)> = HashMap::new();
    for ((rel, _, st, bin), n) in entries.iter().zip(counts) {
        current.insert(rel.as_str(), (*st, n?, bin.as_path()));
    }

    // The joined file: every file's tokens in path order (a project's files
    // stay next to each other). If the files already in it are unchanged,
    // new files are appended (a mapping made earlier keeps seeing its own
    // length); otherwise a new file is written under a new name, so no
    // mapped file is ever replaced.
    let joined = if manifest.joined.is_empty() { "all.bin".to_string() } else { manifest.joined.clone() };
    let expected: u64 = manifest.order.iter().map(|(_, _, n)| 2 * n).sum();
    let reusable = std::fs::metadata(cache.join(&joined)).is_ok_and(|m| m.len() == expected)
        && manifest.order.iter().all(|(rel, st, n)| current.get(rel.as_str()).is_some_and(|c| (c.0, c.1) == (*st, *n)));
    let (joined, mut order, file) = if reusable {
        let file = std::fs::OpenOptions::new().append(true).open(cache.join(&joined))?;
        (joined, manifest.order, file)
    } else {
        let mut n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        while cache.join(format!("all-{n:x}.bin")).exists() {
            n += 1;
        }
        let name = format!("all-{n:x}.bin");
        let file = File::create(cache.join(&name))?;
        (name, Vec::new(), file)
    };
    let listed: HashSet<String> = order.iter().map(|(rel, _, _)| rel.clone()).collect();
    let mut out = std::io::BufWriter::with_capacity(1 << 20, file);
    for (rel, _, st, bin) in &entries {
        if listed.contains(rel) {
            continue;
        }
        let bytes = std::fs::read(bin)?;
        out.write_all(&bytes)?;
        order.push((rel.clone(), *st, bytes.len() as u64 / 2));
    }
    out.flush()?;
    drop(out);
    let files = entries.iter().map(|(rel, _, st, _)| (rel.clone(), *st)).collect();
    let manifest = Manifest { format_hash, files, order, joined: joined.clone() };
    write_atomic(&manifest_path, &serde_json::to_vec(&manifest)?)?;
    // Older joined files go once nothing maps them (on Windows a mapped one
    // can't be deleted yet; it is tried again next time).
    for e in std::fs::read_dir(&cache)?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name != joined && name.starts_with("all") && (name.ends_with(".bin") || name.ends_with(".tmp")) {
            let _ = std::fs::remove_file(e.path());
        }
    }
    let f = File::open(cache.join(&joined))?;
    let map = if f.metadata()?.len() == 0 { None } else { Some(unsafe { memmap2::Mmap::map(&f)? }) };
    drop(lock);
    Ok(Tokens { map })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::testing;

    const NO_FIM: Fim = Fim { rate: 0.0, window_chars: 1000 };

    #[test]
    fn code_keeps_its_lines_and_loses_marker_characters() {
        assert_eq!(clean_code(b"a\r\n    b\x01\tc\rd"), "a\n    b\tc\nd");
    }

    #[test]
    fn a_big_corpus_is_sampled_evenly() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let dir = corpus_dir(&s).join("src");
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("a.py"), "apple = 1\n".repeat(1000)).unwrap();
        std::fs::write(dir.join("b.py"), "banana = 2\n".repeat(3000)).unwrap();
        std::fs::write(dir.join("notes.txt"), "not code").unwrap();
        let sample = read_corpus(&s, 20_000);
        let (a, b) = (sample.matches("apple").count(), sample.matches("banana").count());
        assert!(a > 300 && b > 1000 && sample.len() <= 20_000, "{a} {b} {}", sample.len());
        assert!(sample.lines().all(|l| l == "apple = 1" || l == "banana = 2" || l.is_empty()));
        assert_eq!(read_corpus(&s, 10_000_000).matches("banana").count(), 3000);
        assert!(!read_corpus(&s, 10_000_000).contains("not code"));
    }

    #[test]
    fn fill_in_the_middle_documents_hold_the_whole_file() {
        let code = (0..200).map(|i| format!("line_{i} = {i}\n")).collect::<String>();
        let mut fims = 0;
        for i in 0..40 {
            let path = format!("pkg/m{i}.py");
            let docs = file_documents(&path, &code, Fim { rate: 0.5, window_chars: 700 });
            assert_eq!(docs, file_documents(&path, &code, Fim { rate: 0.5, window_chars: 700 }), "deterministic");
            let mut rebuilt = String::new();
            for d in &docs {
                let body = d.strip_prefix(&format!("{FILE}{path}\n")).unwrap().strip_suffix(EOT).unwrap();
                if let Some(rest) = body.strip_prefix(PRE) {
                    fims += 1;
                    let (pre, rest) = rest.split_once(SUF).unwrap();
                    let (suf, mid) = rest.split_once(MID).unwrap();
                    rebuilt += &format!("{pre}{mid}{suf}");
                } else {
                    rebuilt += body;
                }
            }
            assert_eq!(rebuilt, code);
        }
        assert!(fims > 20, "{fims}");
    }

    #[test]
    fn tokens_are_cached_and_follow_file_changes() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let dir = corpus_dir(&s).join("proj");
        std::fs::create_dir_all(dir.join("sub")).unwrap();
        std::fs::write(dir.join("a.py"), "alpha = 1\n").unwrap();
        std::fs::write(dir.join("sub/b.rs"), "fn delta() {}\n").unwrap();

        let tok = Tokenizer::train(&read_corpus(&s, 10_000), 280);
        let text = |t: &Tokens| tok.decode(&t.iter().map(|&i| i as u32).collect::<Vec<_>>());
        let first = corpus_tokens(&tok, &s, NO_FIM).unwrap();
        let (a, b) = (document("a.py", "alpha = 1\n"), document("sub/b.rs", "fn delta() {}\n"));
        assert_eq!(text(&first), format!("{a}{b}"));
        let bins = std::fs::read_dir(s.data_path("brain/token_cache")).unwrap().count();
        assert_eq!(&*corpus_tokens(&tok, &s, NO_FIM).unwrap(), &*first);
        assert_eq!(std::fs::read_dir(s.data_path("brain/token_cache")).unwrap().count(), bins);

        // a new file is appended; a changed one rebuilds everything in order
        std::fs::write(dir.join("z.py"), "zeta = 3\n").unwrap();
        let z = document("z.py", "zeta = 3\n");
        assert_eq!(text(&corpus_tokens(&tok, &s, NO_FIM).unwrap()), format!("{a}{b}{z}"));
        std::fs::write(dir.join("a.py"), "alpha = 22\n").unwrap();
        let a2 = document("a.py", "alpha = 22\n");
        assert_eq!(text(&corpus_tokens(&tok, &s, NO_FIM).unwrap()), format!("{a2}{b}{z}"));
        // a rebuild never touches a file still mapped; old ones go once released
        assert_eq!(text(&first), format!("{a}{b}"));
        drop(first);
        corpus_tokens(&tok, &s, NO_FIM).unwrap();
        let joined = std::fs::read_dir(s.data_path("brain/token_cache")).unwrap()
            .filter(|e| e.as_ref().unwrap().file_name().to_string_lossy().starts_with("all")).count();
        assert_eq!(joined, 1);
    }
}
