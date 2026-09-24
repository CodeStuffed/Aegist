//! The training corpus: every .txt / .md file under data/corpus/ - text you
//! import, Wikipedia dumps, articles the research loop collects - and its
//! tokens, cached per file so a growing corpus stays cheap to reload.
//!
//! All the tokens are joined into one file on disk (brain/token_cache/all.bin)
//! that training memory-maps instead of loading: billions of tokens cost
//! disk space, not RAM, and the operating system keeps the hot parts cached.

use crate::config::Settings;
use crate::tokenizer::Tokenizer;
use crate::util::{fnv1a, write_atomic};
use anyhow::{bail, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub fn corpus_dir(settings: &Settings) -> PathBuf {
    settings.data_path("corpus")
}

fn is_text(p: &Path) -> bool {
    matches!(p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref(), Some("txt" | "md"))
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    for e in entries.flatten() {
        let p = e.path();
        if p.is_dir() {
            walk(&p, out);
        } else if p.is_file() && is_text(&p) {
            out.push(p);
        }
    }
}

/// Every .txt/.md file in `path` (a file or a folder, searched recursively), sorted.
pub fn text_files(path: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if path.is_file() {
        if is_text(path) {
            files.push(path.to_path_buf());
        }
    } else {
        walk(path, &mut files);
    }
    files.sort();
    files
}

pub fn corpus_files(settings: &Settings) -> Vec<PathBuf> {
    let mut files = Vec::new();
    walk(&corpus_dir(settings), &mut files);
    files.sort();
    files
}

/// Copy .txt/.md files from `src` (a file or folder) into the corpus.
pub fn import_texts(src: &Path, settings: &Settings) -> Result<usize> {
    if !src.exists() {
        bail!("{} doesn't exist", src.display());
    }
    let dest_root = corpus_dir(settings).join("imported");
    let pairs: Vec<(PathBuf, PathBuf)> = if src.is_file() {
        vec![(src.to_path_buf(), dest_root.join(src.file_name().unwrap_or_default()))]
    } else {
        let mut files = Vec::new();
        walk(src, &mut files);
        let name = src.canonicalize()?.file_name().map(|n| n.to_owned()).unwrap_or_default();
        files.into_iter().map(|f| {
            let rel = f.strip_prefix(src).unwrap_or(&f).to_path_buf();
            (f, dest_root.join(&name).join(rel))
        }).collect()
    };
    let mut count = 0;
    for (from, to) in pairs.into_iter().filter(|(f, _)| is_text(f)) {
        if let Some(parent) = to.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::copy(&from, &to)?;
        count += 1;
    }
    Ok(count)
}

/// Unwrap hard-wrapped lines (a single newline is just a space) so the model
/// learns sentences, not line lengths. Paragraph breaks survive.
pub fn normalize_text(text: &str) -> String {
    let text = text.replace("\r\n", "\n");
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    for (i, &ch) in chars.iter().enumerate() {
        let ch = if ch == '\n' && chars.get(i.wrapping_sub(1)) != Some(&'\n') && chars.get(i + 1) != Some(&'\n') {
            ' '
        } else if ch == '\t' {
            ' '
        } else {
            ch
        };
        if ch == ' ' && out.ends_with(' ') {
            continue;
        }
        out.push(ch);
    }
    out.trim().to_string()
}

fn read(path: &Path) -> String {
    let bytes = std::fs::read(path).unwrap_or_default();
    normalize_text(&String::from_utf8_lossy(&bytes))
}

/// Up to `max_chars` of the corpus as one string (paragraph break between
/// files). If the corpus is bigger, every file gives the same share of its
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
                // end on a whole word
                let cut = bytes.iter().rposition(|b| b.is_ascii_whitespace()).unwrap_or(0);
                bytes.truncate(cut);
            }
            normalize_text(&String::from_utf8_lossy(&bytes))
        };
        if text.is_empty() {
            continue;
        }
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&text);
    }
    let mut end = max_chars.min(out.len());
    while !out.is_char_boundary(end) {
        end -= 1;
    }
    out.truncate(end);
    out
}

#[derive(Default, Serialize, Deserialize)]
struct Manifest {
    merges_hash: u64,
    /// file -> (size, modified) when its tokens were cached
    files: BTreeMap<String, (u64, u128)>,
    /// the files whose tokens are in all.bin, in order, with their token counts
    #[serde(default)]
    order: Vec<(String, (u64, u128), u64)>,
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
            // while mapped (rebuilds write a new file and rename it).
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
pub fn corpus_tokens(tok: &Tokenizer, settings: &Settings) -> Result<Tokens> {
    assert!(tok.vocab_size() <= u16::MAX as usize + 1);
    let root = corpus_dir(settings);
    let cache = settings.data_path("brain/token_cache");
    std::fs::create_dir_all(&cache)?;
    // one process at a time (e.g. `research` and `train` in two terminals)
    let lock = File::create(cache.join(".lock"))?;
    lock.lock()?;
    let manifest_path = cache.join("manifest.json");
    let merges_hash = fnv1a(serde_json::to_string(&tok.merges)?.as_bytes());
    let manifest: Manifest = std::fs::read(&manifest_path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .filter(|m: &Manifest| m.merges_hash == merges_hash)
        .unwrap_or(Manifest { merges_hash, ..Default::default() });

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
            let ids: Vec<u16> = tok.encode(&read(path)).into_iter().map(|i| i as u16).collect();
            write_atomic(bin, &u16_bytes(&ids))?;
            Ok(ids.len() as u64)
        })
        .collect();
    let mut current: HashMap<&str, ((u64, u128), u64, &Path)> = HashMap::new();
    for ((rel, _, st, bin), n) in entries.iter().zip(counts) {
        current.insert(rel.as_str(), (*st, n?, bin.as_path()));
    }

    // all.bin: every file's tokens plus a paragraph break. If the files
    // already in it are unchanged, new files are appended; otherwise rebuilt.
    let sep = u16_bytes(&tok.encode("\n\n").into_iter().map(|i| i as u16).collect::<Vec<_>>());
    let all = cache.join("all.bin");
    let expected: u64 = manifest.order.iter().map(|(_, _, n)| 2 * n + sep.len() as u64).sum();
    let reusable = std::fs::metadata(&all).is_ok_and(|m| m.len() == expected)
        && manifest.order.iter().all(|(rel, st, n)| current.get(rel.as_str()).is_some_and(|c| (c.0, c.1) == (*st, *n)));
    let mut order = if reusable { manifest.order } else { Vec::new() };
    let listed: HashSet<String> = order.iter().map(|(rel, _, _)| rel.clone()).collect();
    let target = if reusable { all.clone() } else { cache.join("all.tmp") };
    let mut out = std::io::BufWriter::with_capacity(1 << 20, std::fs::OpenOptions::new().create(true).append(true).open(&target)?);
    if !reusable {
        out.get_ref().set_len(0)?;
    }
    for (rel, _, st, bin) in &entries {
        if listed.contains(rel) {
            continue;
        }
        let bytes = std::fs::read(bin)?;
        out.write_all(&bytes)?;
        out.write_all(&sep)?;
        order.push((rel.clone(), *st, bytes.len() as u64 / 2));
    }
    out.flush()?;
    drop(out);
    if !reusable {
        std::fs::rename(&target, &all)?;
    }
    let files = entries.iter().map(|(rel, _, st, _)| (rel.clone(), *st)).collect();
    write_atomic(&manifest_path, &serde_json::to_vec(&Manifest { merges_hash, files, order })?)?;
    let f = File::open(&all)?;
    let map = if f.metadata()?.len() == 0 { None } else { Some(unsafe { memmap2::Mmap::map(&f)? }) };
    drop(lock);
    Ok(Tokens { map })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::testing;

    #[test]
    fn a_big_corpus_is_sampled_evenly() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        std::fs::create_dir_all(corpus_dir(&s)).unwrap();
        std::fs::write(corpus_dir(&s).join("a.txt"), "apple ".repeat(1000)).unwrap();
        std::fs::write(corpus_dir(&s).join("b.txt"), "banana ".repeat(3000)).unwrap();
        let sample = read_corpus(&s, 2000);
        let (a, b) = (sample.matches("apple").count(), sample.matches("banana").count());
        assert!(a > 50 && b > 150 && sample.len() <= 2000, "{a} {b} {}", sample.len());
        assert!(!sample.contains("appl ") && !sample.ends_with("banan"));
        assert_eq!(read_corpus(&s, 1_000_000).matches("banana").count(), 3000);
    }

    #[test]
    fn normalize_unwraps_hard_wrapped_lines() {
        assert_eq!(normalize_text("one\ntwo  three\n\nfour\r\nfive\tsix"), "one two three\n\nfour five six");
    }

    #[test]
    fn import_copies_only_text_and_tokens_are_cached() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(&tmp.path().join("data"));
        let src = tmp.path().join("src");
        std::fs::create_dir_all(src.join("sub")).unwrap();
        std::fs::write(src.join("a.txt"), "alpha beta gamma").unwrap();
        std::fs::write(src.join("sub/b.md"), "delta\nepsilon").unwrap();
        std::fs::write(src.join("c.pdf"), b"%PDF").unwrap();
        assert_eq!(import_texts(&src, &s).unwrap(), 2);
        let names: Vec<_> = corpus_files(&s).iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, vec!["a.txt", "b.md"]);

        let tok = Tokenizer::train(&read_corpus(&s, 10_000), 280);
        let text = |t: &Tokens| tok.decode(&t.iter().map(|&i| i as u32).collect::<Vec<_>>());
        let first = corpus_tokens(&tok, &s).unwrap();
        assert_eq!(text(&first), "alpha beta gamma\n\ndelta epsilon\n\n");
        let bin = std::fs::read_dir(s.data_path("brain/token_cache")).unwrap().count();
        assert_eq!(&*corpus_tokens(&tok, &s).unwrap(), &*first);
        assert_eq!(std::fs::read_dir(s.data_path("brain/token_cache")).unwrap().count(), bin);

        // a new file is appended; a changed one rebuilds everything in order
        std::fs::write(corpus_dir(&s).join("zeta.txt"), "zeta").unwrap();
        assert_eq!(text(&corpus_tokens(&tok, &s).unwrap()), "alpha beta gamma\n\ndelta epsilon\n\nzeta\n\n");
        std::fs::write(corpus_dir(&s).join("imported/src/a.txt"), "alpha").unwrap();
        assert_eq!(text(&corpus_tokens(&tok, &s).unwrap()), "alpha\n\ndelta epsilon\n\nzeta\n\n");
        std::fs::remove_file(corpus_dir(&s).join("zeta.txt")).unwrap();
        assert_eq!(text(&corpus_tokens(&tok, &s).unwrap()), "alpha\n\ndelta epsilon\n\n");
        assert!(import_texts(&tmp.path().join("missing"), &s).is_err());
    }
}
