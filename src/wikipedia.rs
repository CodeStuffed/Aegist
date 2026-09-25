//! Import a Wikipedia dump: `council import-wikipedia <dump>`.
//!
//! Takes the file Wikipedia publishes for download (for English,
//! enwiki-latest-pages-articles-multistream.xml.bz2, about 22 GB), or the
//! same unpacked (.xml). Every article becomes plain text, written to:
//! - the training corpus (data/corpus/wikipedia/part-*.txt), and
//! - the knowledge base, one passage per paragraph, linked to its article.
//!
//! Streams the whole way: memory use stays flat however big the dump is.
//! The multistream .bz2 is made of many small independent pieces, so they
//! are unpacked on every core at once. Stopping (Ctrl-C) keeps everything
//! imported so far, and running it again carries on where it left off.
//!
//! Wikitext is turned into text by hand here: templates, tables, references,
//! files and categories are dropped; links keep their visible words.

use crate::config::Settings;
use crate::corpus::corpus_dir;
use crate::knowledge::KnowledgeStore;
use crate::trainer::{commas, human_duration};
use crate::util::write_atomic;
use anyhow::{bail, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

/// Compressed bytes read (and unpacked in parallel) at a time.
const READ_BYTES: usize = 64 << 20;
/// Start a new corpus file after this many bytes.
const PART_BYTES: u64 = 64 << 20;
/// Articles converted in parallel at a time.
const BATCH: usize = 2_000;
/// Sections that are lists of sources, not content.
const SKIP_SECTIONS: &[&str] = &[
    "references", "external links", "see also", "notes", "further reading", "bibliography", "sources", "citations",
    "footnotes", "notes and references", "references and notes", "works cited", "general references", "explanatory notes",
];

pub struct ImportOptions {
    /// Stop after this many articles in total (across runs).
    pub max_articles: Option<u64>,
    /// Also store paragraphs in the knowledge base.
    pub knowledge: bool,
}

#[derive(Debug, Default)]
pub struct ImportSummary {
    pub articles: u64,
    pub passages: u64,
    pub corpus_bytes: u64,
    pub interrupted: bool,
}

/// Progress, so a second run with the same dump continues instead of starting over.
#[derive(Default, Serialize, Deserialize)]
struct State {
    dump: String,
    dump_bytes: u64,
    articles_done: u64,
    next_part: u32,
}

fn state_path(settings: &Settings) -> PathBuf {
    corpus_dir(settings).join("wikipedia").join("import-state.json")
}

// ------------------------------------------------------------ reading dumps

fn is_stream_start(b: &[u8]) -> bool {
    // "BZh" + block size digit + the first block's magic number (pi in BCD)
    b.len() >= 10 && &b[..3] == b"BZh" && (b'1'..=b'9').contains(&b[3]) && b[4..10] == [0x31, 0x41, 0x59, 0x26, 0x53, 0x59]
}

fn unpack(streams: &[u8]) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(streams.len() * 5);
    bzip2::read::MultiBzDecoder::new(streams).read_to_end(&mut out)?;
    Ok(out)
}

/// Feed the dump's XML to `sink` in order, a piece at a time; `sink`
/// returns false to stop early. `progress` gets compressed bytes consumed.
fn read_dump(path: &Path, sink: &mut dyn FnMut(&[u8]) -> Result<bool>, progress: &mut dyn FnMut(u64)) -> Result<()> {
    let mut f = File::open(path)?;
    let is_bz2 = path.extension().is_some_and(|e| e.eq_ignore_ascii_case("bz2"));
    let mut buf = vec![0u8; READ_BYTES];
    let mut consumed = 0u64;
    if !is_bz2 {
        loop {
            let n = f.read(&mut buf)?;
            consumed += n as u64;
            progress(consumed);
            if n == 0 || !sink(&buf[..n])? {
                return Ok(());
            }
        }
    }
    // Multistream: cut the compressed bytes where streams start and unpack
    // the pieces on all cores. A single-stream file just unpacks in order.
    let mut carry: Vec<u8> = Vec::new();
    let pieces = rayon::current_num_threads().max(1) * 4;
    loop {
        let n = f.read(&mut buf)?;
        carry.extend_from_slice(&buf[..n]);
        let eof = n == 0;
        let mut starts: Vec<usize> = (0..carry.len().saturating_sub(9)).filter(|&i| carry[i] == b'B' && is_stream_start(&carry[i..])).collect();
        if starts.first() != Some(&0) {
            if carry.is_empty() {
                return Ok(());
            }
            bail!("{} doesn't look like a .bz2 file", path.display());
        }
        if !eof && starts.len() < 2 && carry.len() > 4 * READ_BYTES {
            // one huge stream (not a multistream dump): unpack it in order
            let rest = std::io::Cursor::new(std::mem::take(&mut carry)).chain(f);
            let mut dec = bzip2::read::MultiBzDecoder::new(rest);
            let mut out = vec![0u8; 8 << 20];
            loop {
                let n = dec.read(&mut out)?;
                if n == 0 || !sink(&out[..n])? {
                    return Ok(());
                }
            }
        }
        // the last stream may be incomplete unless the file ended
        let complete = if eof { carry.len() } else { *starts.last().unwrap() };
        starts.retain(|&s| s < complete);
        if !starts.is_empty() && complete > 0 {
            // group streams into about `pieces` jobs
            let per = complete.div_ceil(pieces).max(1);
            let mut cuts = vec![0usize];
            for &s in &starts[1..] {
                if s - cuts.last().unwrap() >= per {
                    cuts.push(s);
                }
            }
            cuts.push(complete);
            let jobs: Vec<&[u8]> = cuts.windows(2).map(|w| &carry[w[0]..w[1]]).collect();
            let unpacked: Vec<Result<Vec<u8>>> = jobs.par_iter().map(|j| unpack(j)).collect();
            for u in unpacked {
                if !sink(&u?)? {
                    return Ok(());
                }
            }
        }
        consumed += complete as u64;
        progress(consumed);
        carry.drain(..complete);
        if eof {
            return Ok(());
        }
    }
}

/// One article as it appears in the dump.
#[derive(Debug, Default)]
pub struct RawPage {
    pub title: String,
    pub wikitext: String,
}

/// Pulls pages out of the XML stream, line by line (the dump puts each tag on
/// its own line, and the text runs over many).
#[derive(Default)]
struct PageParser {
    line: Vec<u8>,
    in_text: bool,
    title: String,
    ns: i64,
    redirect: bool,
    text: String,
    /// e.g. https://en.wikipedia.org/wiki/ (from the dump's <siteinfo>)
    base: Option<String>,
    /// namespace names other than articles, lowercase (from <siteinfo>)
    namespaces: Vec<String>,
}

fn between<'a>(line: &'a str, open: &str, close: &str) -> Option<&'a str> {
    let s = line.find(open)? + open.len();
    let e = line[s..].find(close)? + s;
    Some(&line[s..e])
}

impl PageParser {
    fn feed(&mut self, data: &[u8], out: &mut Vec<RawPage>) {
        let mut rest = data;
        while let Some(i) = rest.iter().position(|&b| b == b'\n') {
            self.line.extend_from_slice(&rest[..i]);
            let line = std::mem::take(&mut self.line);
            self.handle(&String::from_utf8_lossy(&line), out);
            rest = &rest[i + 1..];
        }
        self.line.extend_from_slice(rest);
    }

    fn handle(&mut self, line: &str, out: &mut Vec<RawPage>) {
        if self.in_text {
            match line.find("</text>") {
                Some(e) => {
                    self.text.push_str(&line[..e]);
                    self.in_text = false;
                }
                None => {
                    self.text.push_str(line);
                    self.text.push('\n');
                }
            }
            return;
        }
        let t = line.trim_start();
        if t.starts_with("<page>") {
            (self.title, self.ns, self.redirect) = (String::new(), 0, false);
            self.text.clear();
        } else if let Some(title) = between(t, "<title>", "</title>") {
            self.title = unescape_xml(title);
        } else if let Some(ns) = between(t, "<ns>", "</ns>") {
            self.ns = ns.trim().parse().unwrap_or(-1);
        } else if t.starts_with("<redirect") {
            self.redirect = true;
        } else if t.starts_with("<text") {
            let Some(gt) = t.find('>') else { return };
            if t[..gt].ends_with('/') {
                return; // <text ... /> - empty
            }
            let body = &t[gt + 1..];
            match body.find("</text>") {
                Some(e) => self.text.push_str(&body[..e]),
                None => {
                    self.text.push_str(body);
                    self.text.push('\n');
                    self.in_text = true;
                }
            }
        } else if t.starts_with("</page>") {
            if self.ns == 0 && !self.redirect && !self.title.is_empty() {
                out.push(RawPage { title: std::mem::take(&mut self.title), wikitext: std::mem::take(&mut self.text) });
            }
        } else if t.starts_with("<namespace ") && !t.contains("key=\"0\"") {
            if let Some(name) = between(t, ">", "</namespace>") {
                self.namespaces.push(unescape_xml(name).to_lowercase());
            }
        } else if let Some(base) = between(t, "<base>", "</base>") {
            if let Some(i) = base.find("/wiki/") {
                self.base = Some(base[..i + 6].to_string());
            }
        }
    }
}

// ---------------------------------------------------- wikitext -> plain text

pub fn unescape_xml(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    s.replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", "\"").replace("&#039;", "'").replace("&apos;", "'").replace("&amp;", "&")
}

/// HTML entities left in the text (&nbsp;, &ndash;, &#8212;, ...).
fn decode_entities(s: &str) -> String {
    if !s.contains('&') {
        return s.to_string();
    }
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        rest = &rest[i..];
        let end = rest[1..].find(|c: char| c == ';' || c.is_whitespace() || c == '&').map(|e| e + 1);
        let decoded = end.filter(|&e| rest.as_bytes()[e] == b';' && e <= 10).and_then(|e| {
            let name = &rest[1..e];
            let ch = match name {
                "nbsp" | "thinsp" | "ensp" | "emsp" => Some(' '),
                "ndash" => Some('–'),
                "mdash" => Some('—'),
                "amp" => Some('&'),
                "lt" => Some('<'),
                "gt" => Some('>'),
                "quot" => Some('"'),
                "apos" => Some('\''),
                "minus" => Some('−'),
                "times" => Some('×'),
                "deg" => Some('°'),
                "hellip" => Some('…'),
                "shy" | "zwj" | "zwnj" | "lrm" | "rlm" => Some('\u{0}'),
                _ => name
                    .strip_prefix("#x")
                    .or_else(|| name.strip_prefix("#X"))
                    .and_then(|h| u32::from_str_radix(h, 16).ok())
                    .or_else(|| name.strip_prefix('#').and_then(|d| d.parse().ok()))
                    .and_then(char::from_u32),
            }?;
            Some((ch, e))
        });
        match decoded {
            Some((ch, e)) => {
                if ch != '\u{0}' {
                    out.push(ch);
                }
                rest = &rest[e + 1..];
            }
            None => {
                out.push('&');
                rest = &rest[1..];
            }
        }
    }
    out.push_str(rest);
    out
}

/// Split template/link contents on '|' at nesting depth 0.
fn split_top(s: &str) -> Vec<&str> {
    let b = s.as_bytes();
    let (mut depth, mut start, mut parts) = (0i32, 0, Vec::new());
    let mut i = 0;
    while i < b.len() {
        match (b[i], b.get(i + 1)) {
            (b'{', Some(b'{')) | (b'[', Some(b'[')) => {
                depth += 1;
                i += 2;
                continue;
            }
            (b'}', Some(b'}')) | (b']', Some(b']')) => {
                depth -= 1;
                i += 2;
                continue;
            }
            (b'|', _) if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }
    parts.push(&s[start..]);
    parts
}

/// The few templates that stand for words in a sentence; all others
/// (infoboxes, citations, navigation boxes...) are dropped.
fn template_text(inner: &str) -> String {
    let parts = split_top(inner);
    let name = parts[0].trim().to_lowercase().replace('_', " ");
    let args: Vec<&str> = parts[1..].iter().map(|p| p.trim()).filter(|p| !p.contains('=')).collect();
    let arg = |i: usize| args.get(i).copied().unwrap_or("");
    match name.as_str() {
        "convert" | "cvt" => format!("{} {}", arg(0), arg(1)).trim().to_string(),
        "nowrap" | "nobr" | "small" | "smaller" | "big" | "em" | "strong" | "sic" | "abbr" | "ill" | "lang-rtl" | "mvar" | "math"
        | "var" | "nobold" | "noitalic" | "vr" | "tooltip" | "keypress" | "wikt-lang" => arg(0).to_string(),
        "lang" | "transl" | "transliteration" => args.last().copied().unwrap_or("").to_string(),
        "frac" if args.len() >= 2 => format!("{}/{}", arg(args.len() - 2), arg(args.len() - 1)),
        "circa" | "c." => format!("c. {}", arg(0)),
        "birth date" | "death date" | "start date" | "end date" | "birth date and age" | "death date and age" | "film date" => {
            args.iter().take(3).filter(|a| a.chars().all(|c| c.is_ascii_digit()) && !a.is_empty()).copied().collect::<Vec<_>>().join("-")
        }
        n if n.starts_with("lang-") => arg(0).to_string(),
        _ => String::new(),
    }
}

/// Only indentation (spaces, ':') between the start of the line and `i`.
fn at_line_start(b: &[u8], i: usize) -> bool {
    b[..i].iter().rev().take_while(|&&c| c != b'\n').all(|&c| c == b' ' || c == b':' || c == b'\t')
}

/// Remove {{templates}} (turning a few into words) and {| tables |}.
fn strip_templates(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    let mut plain = 0; // start of text not yet copied
    while i < b.len() {
        let table = b[i] == b'{' && b.get(i + 1) == Some(&b'|') && at_line_start(b, i);
        if (b[i] == b'{' && b.get(i + 1) == Some(&b'{')) || table {
            out.push_str(&s[plain..i]);
            // find the matching close, counting nested templates and tables
            // (table markers only count at the start of a line)
            let (mut templates, mut tables, mut j) = (0i32, 0i32, i);
            while j < b.len() {
                let line_start = at_line_start(b, j);
                if b[j] == b'{' && b.get(j + 1) == Some(&b'{') {
                    templates += 1;
                    j += 2;
                } else if line_start && b[j] == b'{' && b.get(j + 1) == Some(&b'|') {
                    tables += 1;
                    j += 2;
                } else if b[j] == b'}' && b.get(j + 1) == Some(&b'}') && templates > 0 {
                    templates -= 1;
                    j += 2;
                    if templates == 0 && tables == 0 {
                        break;
                    }
                } else if line_start && b[j] == b'|' && b.get(j + 1) == Some(&b'}') && tables > 0 {
                    tables -= 1;
                    j += 2;
                    if templates == 0 && tables == 0 {
                        break;
                    }
                } else {
                    j += 1;
                }
            }
            if !table && templates == 0 && tables == 0 {
                out.push_str(&template_text(&s[i + 2..j - 2]));
            }
            i = j;
            plain = j;
        } else {
            i += 1;
        }
    }
    out.push_str(&s[plain.min(s.len())..]);
    out
}

/// Drop <tag>...</tag> blocks whose content isn't prose, and <ref .../>.
fn strip_blocks(s: &str) -> String {
    const DROP: &[&str] = &["ref", "math", "gallery", "timeline", "score", "syntaxhighlight", "source", "imagemap", "templatedata",
        "mapframe", "graph", "chem", "ce", "references", "hiero", "table", "categorytree", "inputbox", "charinsert"];
    let mut out = String::with_capacity(s.len());
    let lower = s.to_ascii_lowercase();
    let mut i = 0;
    while let Some(off) = lower[i..].find('<') {
        let at = i + off;
        out.push_str(&s[i..at]);
        let rest = &lower[at + 1..];
        let name: String = rest.chars().take_while(|c| c.is_ascii_alphanumeric()).collect();
        if !name.is_empty() && DROP.contains(&name.as_str()) {
            let tag_end = rest.find('>').map(|e| at + 1 + e);
            match tag_end {
                Some(e) if s.as_bytes()[e - 1] == b'/' => i = e + 1, // <ref name="x" />
                Some(e) => {
                    let close = format!("</{name}");
                    i = match lower[e..].find(&close) {
                        Some(c) => lower[e + c..].find('>').map_or(s.len(), |g| e + c + g + 1),
                        None => s.len(),
                    };
                }
                None => i = s.len(),
            }
        } else if rest.starts_with('/') || rest.starts_with(|c: char| c.is_ascii_alphabetic()) {
            // any other tag: keep what it wraps, drop the tag itself
            i = rest.find('>').map_or(at + 1, |e| at + 1 + e + 1);
            if rest[..rest.find('>').unwrap_or(0)].starts_with("br") {
                out.push(' ');
            }
        } else {
            out.push('<');
            i = at + 1;
        }
    }
    out.push_str(&s[i..]);
    out
}

fn is_non_prose_link(target: &str, namespaces: &[String]) -> bool {
    let t = target.trim_start_matches(':').trim();
    let Some(colon) = t.find(':') else { return false };
    let prefix = t[..colon].trim();
    let ns = prefix.to_lowercase();
    namespaces.contains(&ns)
        || matches!(ns.as_str(), "file" | "image" | "category" | "media" | "wikipedia" | "wp" | "help" | "template" | "portal" | "draft" | "module" | "special" | "user" | "talk")
        // another language's Wikipedia: [[de:Brechung]] (prefixes are written in lowercase)
        || ((2..=3).contains(&prefix.len()) && prefix.chars().all(|c| c.is_ascii_lowercase()) && !matches!(prefix, "wikt" | "s" | "b" | "q" | "n" | "v" | "c"))
}

/// [[target|words]] -> words, [[target]] -> target, [[File:...]] -> nothing,
/// [http://... words] -> words.
fn strip_links(s: &str, namespaces: &[String]) -> String {
    let b = s.as_bytes();
    let mut out = String::with_capacity(s.len());
    let (mut i, mut plain) = (0, 0);
    while i < b.len() {
        if b[i] == b'[' && b.get(i + 1) == Some(&b'[') {
            out.push_str(&s[plain..i]);
            let (mut depth, mut j) = (0i32, i);
            while j < b.len() {
                if b[j] == b'[' && b.get(j + 1) == Some(&b'[') {
                    depth += 1;
                    j += 2;
                } else if b[j] == b']' && b.get(j + 1) == Some(&b']') {
                    depth -= 1;
                    j += 2;
                    if depth == 0 {
                        break;
                    }
                } else {
                    j += 1;
                }
            }
            if depth == 0 {
                let inner = &s[i + 2..j - 2];
                let parts = split_top(inner);
                if !is_non_prose_link(parts[0], namespaces) {
                    let shown = if parts.len() > 1 { parts[parts.len() - 1] } else { parts[0].trim_start_matches(':') };
                    out.push_str(&strip_links(shown, namespaces));
                }
            }
            i = j;
            plain = j;
        } else if b[i] == b'[' && (s[i + 1..].starts_with("http") || s[i + 1..].starts_with("//")) {
            out.push_str(&s[plain..i]);
            let end = s[i..].find([']', '\n']).map_or(s.len(), |e| i + e);
            if let Some(sp) = s[i..end].find(' ') {
                out.push_str(&s[i + sp + 1..end]);
            }
            i = if end < s.len() && b[end] == b']' { end + 1 } else { end };
            plain = i;
        } else {
            i += 1;
        }
    }
    out.push_str(&s[plain.min(s.len())..]);
    out
}

fn tidy(line: &str) -> String {
    let mut s = line.replace("'''", "").replace("''", "");
    // leftovers of removed templates: "( )", "(; )", "(, born 1950)" -> "(born 1950)"
    for (from, to) in [("(;", "("), ("(,", "("), ("( ", "("), (" )", ")"), ("()", "")] {
        while s.contains(from) {
            s = s.replace(from, to);
        }
    }
    // magic words like __NOTOC__
    while let Some(a) = s.find("__") {
        match s[a + 2..].find("__") {
            Some(b) if s[a + 2..a + 2 + b].chars().all(|c| c.is_ascii_uppercase()) && b > 0 => s.replace_range(a..a + 4 + b, ""),
            _ => break,
        }
    }
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    // "see the notes or ." -> "see the notes or." (but ".NET" keeps its space)
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len());
    for (i, &c) in chars.iter().enumerate() {
        let (next, after) = (chars.get(i + 1).copied(), chars.get(i + 2).copied());
        if c == ' ' && matches!(next, Some(',' | '.' | ';' | ':')) && after.is_none_or(char::is_whitespace) {
            continue;
        }
        out.push(c);
    }
    out
}

/// Wikitext (as stored in the dump, XML-escaped) -> paragraphs of plain text.
/// `namespaces`: the wiki's namespace names in lowercase ("kategorie", ...),
/// from the dump's header; links into them aren't prose.
pub fn to_paragraphs(escaped: &str, namespaces: &[String]) -> Vec<String> {
    let raw = unescape_xml(escaped);
    let lower = raw.to_lowercase();
    if ["{{disambiguation", "{{disambig}}", "{{dab}}", "{{hndis", "{{geodis", "{{set index article", "{{surname}}", "{{given name}}"]
        .iter()
        .any(|m| lower.contains(m))
    {
        return Vec::new(); // a list of other articles, not an article
    }
    let mut s = raw;
    while let Some(a) = s.find("<!--") {
        let b = s[a..].find("-->").map_or(s.len(), |e| a + e + 3);
        s.replace_range(a..b, "");
    }
    let s = strip_links(&strip_blocks(&strip_templates(&s)), namespaces);
    let s = decode_entities(&s);

    let mut paragraphs = Vec::new();
    let mut current: Vec<String> = Vec::new();
    let mut skip_level: Option<usize> = None;
    let flush = |current: &mut Vec<String>, paragraphs: &mut Vec<String>| {
        let p = tidy(&current.join(" "));
        if p.chars().filter(|c| c.is_alphabetic()).count() >= 3 {
            paragraphs.push(p);
        }
        current.clear();
    };
    for line in s.lines() {
        let t = line.trim();
        if t.starts_with('=') && t.ends_with('=') && t.len() > 2 {
            flush(&mut current, &mut paragraphs);
            let level = t.chars().take_while(|&c| c == '=').count();
            let name = t.trim_matches('=').trim().to_lowercase();
            if skip_level.is_some_and(|l| level <= l) {
                skip_level = None;
            }
            if skip_level.is_none() && SKIP_SECTIONS.contains(&name.as_str()) {
                skip_level = Some(level);
            }
            continue;
        }
        if skip_level.is_some() {
            continue;
        }
        if t.is_empty() {
            flush(&mut current, &mut paragraphs);
            continue;
        }
        if t.starts_with('|') || t.starts_with('!') || t.starts_with("{|") || t.starts_with("|}") || t.starts_with("}}") || t.starts_with("----") {
            continue; // table leftovers, rules
        }
        let item = t.trim_start_matches(['*', '#', ':', ';']).trim();
        if item.len() != t.len() {
            // a list item stands on its own
            flush(&mut current, &mut paragraphs);
            current.push(item.to_string());
            flush(&mut current, &mut paragraphs);
        } else {
            current.push(t.to_string());
        }
    }
    flush(&mut current, &mut paragraphs);
    paragraphs
}

// ------------------------------------------------------------------ import

struct Corpus {
    dir: PathBuf,
    part: u32,
    out: Option<BufWriter<File>>,
    written: u64,
}

impl Corpus {
    fn write(&mut self, text: &str) -> Result<u64> {
        if self.out.is_none() || self.written >= PART_BYTES {
            if let Some(mut f) = self.out.take() {
                f.flush()?;
            }
            self.part += 1;
            self.out = Some(BufWriter::with_capacity(1 << 20, File::create(self.dir.join(format!("part-{:05}.txt", self.part)))?));
            self.written = 0;
        }
        let out = self.out.as_mut().unwrap();
        if self.written > 0 {
            out.write_all(b"\n\n")?;
        }
        out.write_all(text.as_bytes())?;
        self.written += text.len() as u64 + 2;
        Ok(text.len() as u64)
    }
}

pub fn import(dump: &Path, settings: &Settings, opts: &ImportOptions, log: &mut dyn FnMut(String), stop: &AtomicBool) -> Result<ImportSummary> {
    if !dump.is_file() {
        bail!("{} doesn't exist. Download a dump first (see HOW_TO_RUN.md, \"Wikipedia\").", dump.display());
    }
    let dump_bytes = std::fs::metadata(dump)?.len();
    let dir = corpus_dir(settings).join("wikipedia");
    std::fs::create_dir_all(&dir)?;
    let dump_name = dump.file_name().unwrap_or_default().to_string_lossy().into_owned();
    let mut state: State = std::fs::read(state_path(settings)).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    if state.dump != dump_name || state.dump_bytes != dump_bytes {
        if state.articles_done > 0 {
            log(format!("A different dump ({}) was imported before; its text stays, this one is added.", state.dump));
        }
        state = State { dump: dump_name, dump_bytes, articles_done: 0, next_part: state.next_part };
    }
    let skip = state.articles_done;
    if skip > 0 {
        log(format!("Continuing: skipping the {} articles already imported.", commas(skip)));
    }
    if opts.max_articles.is_some_and(|m| skip >= m) {
        log("Already imported that many articles.".into());
        return Ok(ImportSummary::default());
    }

    let mut kb = KnowledgeStore::open(settings)?;
    let min_chars = settings.research.min_passage_chars;
    let mut summary = ImportSummary::default();
    let mut corpus = Corpus { dir, part: state.next_part, out: None, written: 0 };
    let start = Instant::now();
    let mut last_log = 0.0;
    let mut seen = 0u64; // articles in the dump so far, including skipped ones
    let mut parser = PageParser::default();
    let mut pages: Vec<RawPage> = Vec::new();
    let consumed = std::cell::Cell::new(0u64);
    {
        let mut bulk = if opts.knowledge { Some(kb.bulk()?) } else { None };
        let mut done = false;
        let mut process = |pages: &mut Vec<RawPage>, base: &str, namespaces: &[String], summary: &mut ImportSummary, seen: &mut u64| -> Result<bool> {
            let mut batch: Vec<RawPage> = Vec::new();
            for p in pages.drain(..) {
                *seen += 1;
                if *seen > skip && opts.max_articles.is_none_or(|m| *seen <= m) {
                    batch.push(p);
                }
            }
            let converted: Vec<(String, Vec<String>)> = batch.into_par_iter().map(|p| {
                let paras = to_paragraphs(&p.wikitext, namespaces);
                (p.title, paras)
            }).collect();
            for (title, paras) in converted {
                summary.articles += 1;
                if paras.is_empty() {
                    continue;
                }
                summary.corpus_bytes += corpus.write(&format!("{title}\n\n{}", paras.join("\n\n")))?;
                if let Some(b) = bulk.as_mut() {
                    let url = format!("{base}{}", title.replace(' ', "_"));
                    for p in paras.iter().filter(|p| p.len() >= min_chars) {
                        let mut meta = Map::new();
                        for (k, v) in [("title", title.as_str()), ("url", url.as_str()), ("source", "wikipedia-dump")] {
                            meta.insert(k.into(), Value::from(v));
                        }
                        if b.add(p, meta)? {
                            summary.passages += 1;
                        }
                    }
                }
            }
            Ok(opts.max_articles.is_none_or(|m| *seen < m) && !stop.load(Ordering::Relaxed))
        };
        read_dump(
            dump,
            &mut |data| {
                parser.feed(data, &mut pages);
                if pages.len() >= BATCH {
                    let base = parser.base.clone().unwrap_or_else(|| "https://en.wikipedia.org/wiki/".into());
                    if !process(&mut pages, &base, &parser.namespaces.clone(), &mut summary, &mut seen)? {
                        done = true;
                        return Ok(false);
                    }
                }
                let now = start.elapsed().as_secs_f64();
                if now - last_log >= 10.0 {
                    last_log = now;
                    let frac = consumed.get() as f64 / dump_bytes.max(1) as f64;
                    let eta = if frac > 0.001 { format!(", about {} left", human_duration(now / frac - now)) } else { String::new() };
                    log(format!("{:5.1}% of the dump | {} articles imported | {:.0} MB of text | {} passages{eta}",
                        100.0 * frac, commas(summary.articles), summary.corpus_bytes as f64 / 1e6,
                        commas(summary.passages)));
                }
                Ok(true)
            },
            &mut |c| consumed.set(c),
        )?;
        if !done && !pages.is_empty() {
            let base = parser.base.clone().unwrap_or_else(|| "https://en.wikipedia.org/wiki/".into());
            process(&mut pages, &base, &parser.namespaces.clone(), &mut summary, &mut seen)?;
        }
        if let Some(b) = bulk {
            b.finish()?;
        }
    }
    if let Some(mut f) = corpus.out.take() {
        f.flush()?;
    }
    summary.interrupted = stop.load(Ordering::Relaxed);
    state.articles_done = skip + summary.articles;
    state.next_part = corpus.part;
    write_atomic(&state_path(settings), &serde_json::to_vec_pretty(&state)?)?;
    if opts.knowledge && summary.passages > 0 {
        log(format!("Indexing {} new passages for search (this can take a while for a whole dump)...", commas(summary.passages)));
        kb.commit()?;
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::testing;
    use std::io::Write;

    const ARTICLE: &str = r#"{{Short description|Physical phenomenon}}
{{Infobox physics
| name = Refraction
| image = [[File:Prism.png|thumb|A [[prism]] at work]]
}}
'''Refraction''' is the change in direction of a [[wave]] passing from one [[Transmission medium|medium]] to another.<ref name="hecht">{{cite book |last=Hecht |title=Optics}}</ref> It was described by [[Ibn Sahl]] ({{circa|984}}) and later by [[Willebrord Snellius|Snell]].<ref>Smith, 2001.</ref>

[[File:Refraction.svg|thumb|upright=1.2|Light [[refraction|refracting]] at an interface]]
Light slows down in glass by about {{convert|30|percent}} &ndash; a large effect.<!-- check this -->

== Explanation ==
The speed of light in a medium depends on its [[refractive index]], see [https://example.org the notes] or [https://example.org].
{| class="wikitable"
|-
! Medium !! Index
|-
| Water || 1.33
|}
* Water bends light less than glass does.
* Diamond bends it the most.

== See also ==
* [[Snell's law]]

== References ==
{{Reflist}}

[[Category:Optics]]
[[de:Brechung]]
"#;

    #[test]
    fn wikitext_becomes_plain_paragraphs() {
        let escaped = ARTICLE.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;");
        let p = to_paragraphs(&escaped, &[]);
        assert_eq!(p[0], "Refraction is the change in direction of a wave passing from one medium to another. \
                          It was described by Ibn Sahl (c. 984) and later by Snell.");
        assert_eq!(p[1], "Light slows down in glass by about 30 percent – a large effect.");
        assert_eq!(p[2], "The speed of light in a medium depends on its refractive index, see the notes or.");
        assert_eq!(p[3], "Water bends light less than glass does.");
        assert_eq!(p[4], "Diamond bends it the most.");
        assert_eq!(p.len(), 5, "{p:#?}");
        let all = p.join(" ");
        for junk in ["Infobox", "Hecht", "Category", "Brechung", "Snell's law", "wikitable", "Smith", "check this", "{{", "[[", "<ref", "'''"] {
            assert!(!all.contains(junk), "{junk:?} left in {all}");
        }
        assert!(to_paragraphs("'''Mercury''' may refer to:\n* [[Mercury (planet)]]\n{{disambiguation}}", &[]).is_empty());
        let de = to_paragraphs("Ein Satz über [[Werbung]] und mehr.\n\n[[Kategorie:Werbeagentur]] [[Datei:X.png|mini|Bild]]", &["kategorie".into(), "datei".into()]);
        assert_eq!(de, vec!["Ein Satz über Werbung und mehr."]);
        assert_eq!(to_paragraphs("Rankings:\n:{|class=wikitable\n|-\n| A || 1\n|}\nDone here.", &[]), vec!["Rankings:", "Done here."]);
    }

    #[test]
    fn entities_and_tags() {
        assert_eq!(decode_entities("a&nbsp;b &#8212; &#x41; &unknown; & c"), "a b — A &unknown; & c");
        assert_eq!(strip_blocks("x<ref name=a/>y<math>x^2</math>z<sup>2</sup>w<br/>v"), "xyz2w v");
        assert_eq!(strip_links("[[a|b [[c]]]] [[:Category:X]] [[wikt:word|word]] [[Main Page#History]] [[Tom: The Movie]]", &[]),
                   "b c  word Main Page#History Tom: The Movie");
    }

    fn dump_xml(pages: &[(&str, i32, bool, &str)]) -> String {
        let mut x = String::from("<mediawiki>\n  <siteinfo>\n    <base>https://en.wikipedia.org/wiki/Main_Page</base>\n  </siteinfo>\n");
        for (title, ns, redirect, text) in pages {
            let esc = text.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;");
            let title = title.replace('&', "&amp;");
            x += &format!("  <page>\n    <title>{title}</title>\n    <ns>{ns}</ns>\n    <id>1</id>\n");
            if *redirect {
                x += "    <redirect title=\"Other\" />\n";
            }
            x += &format!("    <revision>\n      <text bytes=\"1\" xml:space=\"preserve\">{esc}</text>\n    </revision>\n  </page>\n");
        }
        x + "</mediawiki>\n"
    }

    fn bz2(parts: &[String]) -> Vec<u8> {
        // a multistream file: independent streams back to back
        parts.iter().flat_map(|p| {
            let mut e = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::best());
            e.write_all(p.as_bytes()).unwrap();
            e.finish().unwrap()
        }).collect()
    }

    #[test]
    fn imports_a_multistream_dump_and_resumes() {
        let tmp = tempfile::tempdir().unwrap();
        let mut s = testing::settings(&tmp.path().join("data"));
        s.research.min_passage_chars = 50;
        let para = |w: &str| format!("{w} is a word that this paragraph repeats so that it is long enough to be stored as a passage in the base.");
        let a = format!("{}\n\n{}\n\n== References ==\n{{{{Reflist}}}}", para("Alpha"), para("Beta"));
        let b = format!("{}\n\n{}", para("Gamma"), para("Delta"));
        let xml = dump_xml(&[("Alpha article", 0, false, &a), ("Talk:Alpha", 1, false, &a), ("Old name", 0, true, "#REDIRECT [[Alpha]]"),
                             ("Beta & co", 0, false, &b), ("Third", 0, false, &para("Epsilon"))]);
        // cut the XML in three arbitrary places, one stream each
        let cuts = [0, xml.len() / 3, xml.len() / 2 + 7, xml.len()];
        let parts: Vec<String> = cuts.windows(2).map(|w| xml[w[0]..w[1]].to_string()).collect();
        let dump = tmp.path().join("enwiki-test-pages-articles-multistream.xml.bz2");
        std::fs::write(&dump, bz2(&parts)).unwrap();

        let stop = AtomicBool::new(false);
        let opts = ImportOptions { max_articles: Some(2), knowledge: true };
        let r = import(&dump, &s, &opts, &mut |_| {}, &stop).unwrap();
        assert_eq!((r.articles, r.passages), (2, 4));
        let kb = KnowledgeStore::open(&s).unwrap();
        assert_eq!(kb.count(), 4);
        assert_eq!(kb.stats().1, 4); // indexed on disk
        let hit = &kb.search("gamma word paragraph", 1, 0.0)[0];
        assert_eq!(hit.passage.meta("title"), "Beta & co");
        assert_eq!(hit.passage.meta("url"), "https://en.wikipedia.org/wiki/Beta_&_co");
        let linked = kb.search_linked("gamma", 1, 0.1, 10_000);
        assert!(linked.iter().any(|h| h.via.contains("next to it in Beta & co")));

        // the rest, then nothing more
        let all = ImportOptions { max_articles: None, knowledge: true };
        let r = import(&dump, &s, &all, &mut |_| {}, &stop).unwrap();
        assert_eq!((r.articles, r.passages), (1, 1));
        let r = import(&dump, &s, &all, &mut |_| {}, &stop).unwrap();
        assert_eq!((r.articles, r.passages), (0, 0));
        let text: String = crate::corpus::corpus_files(&s).iter().map(|f| std::fs::read_to_string(f).unwrap()).collect::<Vec<_>>().join("|");
        assert!(text.starts_with("Alpha article\n\nAlpha is a word"));
        assert!(text.contains("Beta & co\n\nGamma is") && text.contains("Third\n\nEpsilon"));
        assert!(!text.contains("Reflist") && !text.contains("REDIRECT") && !text.contains("Talk:"));

        // the same XML, unpacked, reads the same
        let tmp2 = tempfile::tempdir().unwrap();
        let s2 = testing::settings(&tmp2.path().join("data"));
        let xml_path = tmp2.path().join("dump.xml");
        std::fs::write(&xml_path, &xml).unwrap();
        let r = import(&xml_path, &s2, &ImportOptions { max_articles: None, knowledge: false }, &mut |_| {}, &stop).unwrap();
        assert_eq!((r.articles, r.passages), (3, 0));
        assert!(import(&tmp2.path().join("missing.xml.bz2"), &s2, &all, &mut |_| {}, &stop).is_err());
    }
}
