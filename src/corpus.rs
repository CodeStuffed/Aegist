//! The training corpus: every .txt / .md file under data/corpus/ - text you
//! import, plus articles the research loop collects - and its tokens,
//! cached per file so a growing corpus stays cheap to reload.

use crate::config::Settings;
use crate::tokenizer::Tokenizer;
use crate::util::{fnv1a, write_atomic};
use anyhow::{bail, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
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

/// The corpus as one string (paragraph break between files), capped at `max_chars`.
pub fn read_corpus(settings: &Settings, max_chars: usize) -> String {
    let mut out = String::new();
    for f in corpus_files(settings) {
        if out.len() >= max_chars {
            break;
        }
        if !out.is_empty() {
            out.push_str("\n\n");
        }
        out.push_str(&read(&f));
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
    files: BTreeMap<String, (u64, u128)>,
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
pub fn corpus_tokens(tok: &Tokenizer, settings: &Settings) -> Result<Vec<u16>> {
    assert!(tok.vocab_size() <= u16::MAX as usize + 1);
    let root = corpus_dir(settings);
    let cache = settings.data_path("brain/token_cache");
    let manifest_path = cache.join("manifest.json");
    let merges_hash = fnv1a(serde_json::to_string(&tok.merges)?.as_bytes());
    let mut manifest: Manifest = std::fs::read(&manifest_path)
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .filter(|m: &Manifest| m.merges_hash == merges_hash)
        .unwrap_or(Manifest { merges_hash, files: BTreeMap::new() });

    let files = corpus_files(settings);
    let entries: Vec<(String, PathBuf, (u64, u128), PathBuf)> = files
        .iter()
        .map(|f| {
            let rel = f.strip_prefix(&root).unwrap_or(f).to_string_lossy().replace('\\', "/");
            let bin = cache.join(format!("{:016x}.bin", fnv1a(rel.as_bytes())));
            (rel, f.clone(), stamp(f), bin)
        })
        .collect();
    let encoded: Vec<Result<Vec<u16>>> = entries
        .par_iter()
        .map(|(rel, path, st, bin)| {
            if manifest.files.get(rel) == Some(st) {
                if let Ok(bytes) = std::fs::read(bin) {
                    return Ok(bytes.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect());
                }
            }
            let ids: Vec<u16> = tok.encode(&read(path)).into_iter().map(|i| i as u16).collect();
            let bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
            write_atomic(bin, &bytes)?;
            Ok(ids)
        })
        .collect();
    let sep: Vec<u16> = tok.encode("\n\n").into_iter().map(|i| i as u16).collect();
    let mut all = Vec::new();
    manifest.files.clear();
    for ((rel, _, st, _), ids) in entries.into_iter().zip(encoded) {
        all.extend(ids?);
        all.extend_from_slice(&sep);
        manifest.files.insert(rel, st);
    }
    write_atomic(&manifest_path, &serde_json::to_vec(&manifest)?)?;
    Ok(all)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::testing;

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
        let first = corpus_tokens(&tok, &s).unwrap();
        assert_eq!(tok.decode(&first.iter().map(|&i| i as u32).collect::<Vec<_>>()), "alpha beta gamma\n\ndelta epsilon\n\n");
        let bin = std::fs::read_dir(s.data_path("brain/token_cache")).unwrap().count();
        assert_eq!(corpus_tokens(&tok, &s).unwrap(), first);
        assert_eq!(std::fs::read_dir(s.data_path("brain/token_cache")).unwrap().count(), bin);
        assert!(import_texts(&tmp.path().join("missing"), &s).is_err());
    }
}
