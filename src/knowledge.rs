//! The local knowledge base: passages of text with metadata, searchable with
//! BM25 (the classic ranking function behind keyword search engines),
//! implemented here from scratch - no embedding model, no outside AI.
//!
//! Passages are stored one JSON object per line in
//! data/knowledge_base/passages.jsonl. The search index sits next to it in
//! index/, on disk, as segments that are memory-mapped rather than loaded: a
//! knowledge base the size of Wikipedia opens instantly and only costs RAM
//! for the parts a search actually touches. Passages added since the last
//! `commit` are indexed in memory until then.
//!
//! Passages are also linked, like notes in Obsidian: to the paragraphs next
//! to them in the same article, and to passages sharing their rarest words.
//! `search_linked` finds the best matches, then follows those links to fill a
//! fixed text budget - the model's context window is small, so it gets the
//! relevant passage plus its closest context, and nothing else.

use crate::config::Settings;
use crate::text::content_words;
use crate::util::{fnv1a, write_atomic};
use anyhow::{Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const K1: f64 = 1.5;
const B: f64 = 0.75;
/// Links weaker than this (link strength x relevance of the source) aren't followed.
const LINK_MIN: f64 = 0.15;
/// `commit_if_large` moves the in-memory passages to disk past this many.
const TAIL_COMMIT: usize = 5_000;
/// Opening indexes on disk first when more passage text than this isn't
/// (a fresh import, an old index version): never load it all into RAM.
const MAX_UNINDEXED_BYTES: u64 = 256 << 20;
/// Postings held in memory while building a segment before spilling a sorted
/// run to disk (~8 bytes each, plus the words): bounds RAM at any size.
const CHUNK_POSTINGS: usize = 30_000_000;
const INDEX_VERSION: u32 = 2;
/// A title word counts as this many occurrences in each of the article's
/// passages: paragraphs rarely repeat their article's name.
const TITLE_WEIGHT: u32 = 2;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Passage {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

impl Passage {
    pub fn meta(&self, key: &str) -> &str {
        self.metadata.get(key).and_then(|v| v.as_str()).unwrap_or("")
    }

    /// The article it came from (title, else url), if known.
    fn article(&self) -> Option<String> {
        [self.meta("title"), self.meta("url")].into_iter().find(|s| !s.is_empty()).map(str::to_string)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Hit {
    #[serde(flatten)]
    pub passage: Passage,
    pub score: f64,
    /// Score as a fraction of the best this query could get (for a linked
    /// passage: its link strength times the relevance of what linked to it).
    pub relevance: f64,
    /// How it was found: "match", or the link that led to it.
    pub via: String,
}

fn normalize(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// What a passage is indexed under: its words, plus its title's.
fn index_terms(p: &Passage) -> HashMap<String, u32> {
    let mut counts = term_counts(&p.text);
    for w in content_words(p.meta("title")) {
        *counts.entry(w).or_default() += TITLE_WEIGHT;
    }
    counts
}

/// Word -> how often it occurs, for one passage.
fn term_counts(text: &str) -> HashMap<String, u32> {
    let mut counts: HashMap<String, u32> = HashMap::new();
    for w in content_words(text) {
        *counts.entry(w).or_default() += 1;
    }
    counts
}

// ---------------------------------------------------------------- on disk

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct SegMeta {
    dir: String,
    /// passages [start_doc, end_doc), stored at bytes [start_byte, end_byte) of passages.jsonl
    start_doc: u32,
    end_doc: u32,
    start_byte: u64,
    end_byte: u64,
    /// total word count, for BM25's average passage length
    total_len: u64,
}

#[derive(Default, Serialize, Deserialize)]
struct Manifest {
    version: u32,
    segments: Vec<SegMeta>,
}

fn read_manifest(dir: &Path) -> Manifest {
    std::fs::read(dir.join("manifest.json"))
        .ok()
        .and_then(|b| serde_json::from_slice::<Manifest>(&b).ok())
        .filter(|m| m.version == INDEX_VERSION)
        .unwrap_or_default()
}

enum Bytes {
    Map(memmap2::Mmap),
    Empty,
}

impl std::ops::Deref for Bytes {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            Bytes::Map(m) => m,
            Bytes::Empty => &[],
        }
    }
}

fn map(path: &Path) -> Result<Bytes> {
    let f = File::open(path).with_context(|| format!("opening {}", path.display()))?;
    if f.metadata()?.len() == 0 {
        return Ok(Bytes::Empty);
    }
    // Safety: segment files are written once, then only ever read or deleted.
    Ok(Bytes::Map(unsafe { memmap2::Mmap::map(&f)? }))
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}

fn u64_at(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

fn put_varint(out: &mut Vec<u8>, mut v: u32) {
    while v >= 0x80 {
        out.push(v as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn get_varint(b: &[u8], at: &mut usize) -> u32 {
    let (mut v, mut shift) = (0u32, 0);
    loop {
        let byte = b[*at];
        *at += 1;
        v |= ((byte & 0x7f) as u32) << shift;
        if byte < 0x80 {
            return v;
        }
        shift += 7;
    }
}

/// (doc, count) pairs of one word in one segment, delta- and varint-encoded.
struct Postings<'a> {
    bytes: &'a [u8],
    at: usize,
    doc: u32,
}

impl Iterator for Postings<'_> {
    type Item = (u32, u32);
    fn next(&mut self) -> Option<(u32, u32)> {
        if self.at >= self.bytes.len() {
            return None;
        }
        self.doc += get_varint(self.bytes, &mut self.at);
        Some((self.doc, get_varint(self.bytes, &mut self.at)))
    }
}

// fixed-size records
const LEX: usize = 32; // word offset u64, word length u32, passages containing it u32, postings offset u64, postings length u64
const DOC: usize = 28; // line offset u64, line length u32, text length u32, word count u32, article u32, position in article u32
const ART: usize = 12; // first entry in art_docs u64, count u32
const NO_ARTICLE: u32 = u32::MAX;

struct DocRec {
    off: u64,
    line_len: u32,
    text_len: u32,
    length: u32,
    article: u32,
    pos: u32,
}

/// One immutable, memory-mapped piece of the index.
struct Segment {
    meta: SegMeta,
    lex: Bytes,
    words: Bytes,
    postings: Bytes,
    docs: Bytes,
    articles: Bytes,
    art_docs: Bytes,
    hashes: Bytes,
}

impl Segment {
    fn open(dir: &Path, meta: SegMeta) -> Result<Segment> {
        let d = dir.join(&meta.dir);
        Ok(Segment {
            lex: map(&d.join("lex.bin"))?,
            words: map(&d.join("words.bin"))?,
            postings: map(&d.join("postings.bin"))?,
            docs: map(&d.join("docs.bin"))?,
            articles: map(&d.join("articles.bin"))?,
            art_docs: map(&d.join("art_docs.bin"))?,
            hashes: map(&d.join("hashes.bin"))?,
            meta,
        })
    }

    fn word(&self, i: usize) -> &[u8] {
        let r = i * LEX;
        let off = u64_at(&self.lex, r) as usize;
        &self.words[off..off + u32_at(&self.lex, r + 8) as usize]
    }

    /// Binary search of the sorted word list.
    fn lookup(&self, term: &str) -> Option<(u32, Postings<'_>)> {
        let (mut lo, mut hi) = (0, self.lex.len() / LEX);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match self.word(mid).cmp(term.as_bytes()) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => {
                    let r = mid * LEX;
                    let off = u64_at(&self.lex, r + 16) as usize;
                    let len = u64_at(&self.lex, r + 24) as usize;
                    let bytes = &self.postings[off..off + len];
                    return Some((u32_at(&self.lex, r + 12), Postings { bytes, at: 0, doc: self.meta.start_doc }));
                }
            }
        }
        None
    }

    fn doc(&self, doc: u32) -> DocRec {
        let r = (doc - self.meta.start_doc) as usize * DOC;
        let b = &self.docs;
        DocRec {
            off: u64_at(b, r),
            line_len: u32_at(b, r + 8),
            text_len: u32_at(b, r + 12),
            length: u32_at(b, r + 16),
            article: u32_at(b, r + 20),
            pos: u32_at(b, r + 24),
        }
    }

    fn article_docs(&self, article: u32) -> Vec<u32> {
        let r = article as usize * ART;
        let first = u64_at(&self.articles, r) as usize;
        let n = u32_at(&self.articles, r + 8) as usize;
        (first..first + n).map(|i| u32_at(&self.art_docs, i * 4)).collect()
    }

    fn contains_hash(&self, h: u64) -> bool {
        let (mut lo, mut hi) = (0, self.hashes.len() / 8);
        while lo < hi {
            let mid = (lo + hi) / 2;
            match u64_at(&self.hashes, mid * 8).cmp(&h) {
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
                std::cmp::Ordering::Equal => return true,
            }
        }
        false
    }
}

fn read_at(f: &File, buf: &mut [u8], off: u64) -> std::io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        f.read_exact_at(buf, off)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            let n = f.seek_read(&mut buf[done..], off + done as u64)?;
            if n == 0 {
                return Err(std::io::ErrorKind::UnexpectedEof.into());
            }
            done += n;
        }
        Ok(())
    }
}

/// Where passages.jsonl's last complete line ends (a writer may be mid-line).
fn complete_len(path: &Path) -> Result<u64> {
    let Ok(mut f) = File::open(path) else { return Ok(0) };
    let len = f.metadata()?.len();
    let mut end = len;
    let mut buf = vec![0u8; 64 * 1024];
    while end > 0 {
        let start = end.saturating_sub(buf.len() as u64);
        let chunk = &mut buf[..(end - start) as usize];
        f.seek(SeekFrom::Start(start))?;
        f.read_exact(chunk)?;
        if let Some(i) = chunk.iter().rposition(|&b| b == b'\n') {
            return Ok(start + i as u64 + 1);
        }
        end = start;
    }
    Ok(0)
}

// --------------------------------------------------------- building segments

struct Parsed {
    off: u64,
    line_len: u32,
    text_len: u32,
    length: u32,
    hash: u64,
    article: Option<String>,
    counts: Vec<(String, u32)>,
}

fn parse_line(line: &[u8], off: u64) -> Option<Parsed> {
    let p: Passage = serde_json::from_slice(line).ok()?;
    let counts: Vec<(String, u32)> = index_terms(&p).into_iter().collect();
    Some(Parsed {
        off,
        line_len: line.len() as u32,
        text_len: p.text.len() as u32,
        length: counts.iter().map(|c| c.1).sum(),
        hash: fnv1a(normalize(&p.text).as_bytes()),
        article: p.article(),
        counts,
    })
}

/// Writes the word list and postings of a segment, one word at a time, in order.
struct LexWriter {
    lex: BufWriter<File>,
    words: BufWriter<File>,
    postings: BufWriter<File>,
    word_off: u64,
    post_off: u64,
    start_doc: u32,
    buf: Vec<u8>,
}

impl LexWriter {
    fn new(dir: &Path, start_doc: u32) -> Result<Self> {
        let w = |name: &str| -> Result<BufWriter<File>> { Ok(BufWriter::new(File::create(dir.join(name))?)) };
        Ok(LexWriter { lex: w("lex.bin")?, words: w("words.bin")?, postings: w("postings.bin")?, word_off: 0, post_off: 0, start_doc, buf: Vec::new() })
    }

    fn write(&mut self, word: &[u8], docs: &[(u32, u32)]) -> Result<()> {
        self.buf.clear();
        let mut prev = self.start_doc;
        for &(doc, tf) in docs {
            put_varint(&mut self.buf, doc - prev);
            put_varint(&mut self.buf, tf);
            prev = doc;
        }
        self.lex.write_all(&self.word_off.to_le_bytes())?;
        self.lex.write_all(&(word.len() as u32).to_le_bytes())?;
        self.lex.write_all(&(docs.len() as u32).to_le_bytes())?;
        self.lex.write_all(&self.post_off.to_le_bytes())?;
        self.lex.write_all(&(self.buf.len() as u64).to_le_bytes())?;
        self.words.write_all(word)?;
        self.postings.write_all(&self.buf)?;
        self.word_off += word.len() as u64;
        self.post_off += self.buf.len() as u64;
        Ok(())
    }

    fn finish(mut self) -> Result<()> {
        self.lex.flush()?;
        self.words.flush()?;
        self.postings.flush()?;
        Ok(())
    }
}

/// A sorted run spilled to disk: (word, [(doc, count)]) records in word order.
fn spill(dir: &Path, n: usize, chunk: &mut HashMap<String, Vec<(u32, u32)>>) -> Result<PathBuf> {
    let path = dir.join(format!("run-{n}.tmp"));
    let mut w = BufWriter::new(File::create(&path)?);
    let mut words: Vec<(String, Vec<(u32, u32)>)> = chunk.drain().collect();
    words.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
    for (word, docs) in words {
        w.write_all(&(word.len() as u32).to_le_bytes())?;
        w.write_all(word.as_bytes())?;
        w.write_all(&(docs.len() as u32).to_le_bytes())?;
        for (d, tf) in docs {
            w.write_all(&d.to_le_bytes())?;
            w.write_all(&tf.to_le_bytes())?;
        }
    }
    w.flush()?;
    Ok(path)
}

fn read_run_record(r: &mut BufReader<File>) -> Result<Option<(Vec<u8>, Vec<(u32, u32)>)>> {
    let mut n4 = [0u8; 4];
    match r.read_exact(&mut n4) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(e) => return Err(e.into()),
    }
    let mut word = vec![0u8; u32::from_le_bytes(n4) as usize];
    r.read_exact(&mut word)?;
    r.read_exact(&mut n4)?;
    let n = u32::from_le_bytes(n4) as usize;
    let mut raw = vec![0u8; n * 8];
    r.read_exact(&mut raw)?;
    let docs = raw.chunks_exact(8).map(|c| (u32_at(c, 0), u32_at(c, 4))).collect();
    Ok(Some((word, docs)))
}

fn write_u32s(path: &Path, v: impl IntoIterator<Item = u32>) -> Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    for x in v {
        w.write_all(&x.to_le_bytes())?;
    }
    w.flush()?;
    Ok(())
}

/// Index the passages at bytes [start_byte, end_byte) of `jsonl` (numbered
/// from `start_doc`) into a new segment folder under `index_dir`. Memory stays
/// bounded however many passages there are: postings are spilled to disk in
/// sorted runs every `chunk_postings`, then merged.
fn build_segment(jsonl: &Path, index_dir: &Path, start_doc: u32, start_byte: u64, end_byte: u64, chunk_postings: usize) -> Result<SegMeta> {
    let nonce = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
    let name = format!("seg-{start_doc:010}-{:x}", fnv1a(format!("{nonce}-{}", std::process::id()).as_bytes()));
    let dir = index_dir.join(&name);
    std::fs::create_dir_all(&dir)?;

    let mut f = File::open(jsonl)?;
    f.seek(SeekFrom::Start(start_byte))?;
    let mut reader = BufReader::with_capacity(1 << 20, f.take(end_byte - start_byte));
    let mut docs_w = BufWriter::new(File::create(dir.join("docs.bin"))?);
    let mut hashes: Vec<u64> = Vec::new();
    let mut art_index: HashMap<String, u32> = HashMap::new();
    let mut art_docs: Vec<Vec<u32>> = Vec::new();
    let mut chunk: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
    let (mut chunk_n, mut runs) = (0usize, Vec::new());
    let (mut doc, mut off, mut total_len) = (start_doc, start_byte, 0u64);
    loop {
        let mut batch: Vec<(u64, Vec<u8>)> = Vec::new();
        while batch.len() < 20_000 {
            let mut line = Vec::new();
            let n = reader.read_until(b'\n', &mut line)?;
            if n == 0 {
                break;
            }
            batch.push((off, line));
            off += n as u64;
        }
        if batch.is_empty() {
            break;
        }
        let parsed: Vec<Option<Parsed>> = batch.par_iter().map(|(o, line)| parse_line(line, *o)).collect();
        for p in parsed.into_iter().flatten() {
            let (article, pos) = match p.article {
                Some(key) => {
                    let next = art_docs.len() as u32;
                    let a = *art_index.entry(key).or_insert(next);
                    if a == next {
                        art_docs.push(Vec::new());
                    }
                    art_docs[a as usize].push(doc);
                    (a, art_docs[a as usize].len() as u32 - 1)
                }
                None => (NO_ARTICLE, 0),
            };
            let mut rec = [0u8; DOC];
            rec[..8].copy_from_slice(&p.off.to_le_bytes());
            for (i, x) in [p.line_len, p.text_len, p.length, article, pos].into_iter().enumerate() {
                rec[8 + 4 * i..12 + 4 * i].copy_from_slice(&x.to_le_bytes());
            }
            docs_w.write_all(&rec)?;
            hashes.push(p.hash);
            total_len += p.length as u64;
            chunk_n += p.counts.len();
            for (t, n) in p.counts {
                chunk.entry(t).or_default().push((doc, n));
            }
            doc += 1;
        }
        if chunk_n >= chunk_postings {
            runs.push(spill(&dir, runs.len(), &mut chunk)?);
            chunk_n = 0;
        }
    }
    docs_w.flush()?;

    let mut lex = LexWriter::new(&dir, start_doc)?;
    if runs.is_empty() {
        let mut words: Vec<(String, Vec<(u32, u32)>)> = chunk.drain().collect();
        words.par_sort_unstable_by(|a, b| a.0.cmp(&b.0));
        for (w, d) in &words {
            lex.write(w.as_bytes(), d)?;
        }
    } else {
        if !chunk.is_empty() {
            runs.push(spill(&dir, runs.len(), &mut chunk)?);
        }
        // k-way merge of the sorted runs; runs hold increasing doc ranges, so
        // a word's postings are the concatenation in run order
        let mut readers: Vec<BufReader<File>> = runs.iter().map(|p| Ok(BufReader::with_capacity(1 << 20, File::open(p)?))).collect::<Result<_>>()?;
        let mut current: Vec<Option<(Vec<u8>, Vec<(u32, u32)>)>> = readers.iter_mut().map(read_run_record).collect::<Result<_>>()?;
        let mut heap: BinaryHeap<std::cmp::Reverse<(Vec<u8>, usize)>> = current
            .iter()
            .enumerate()
            .filter_map(|(i, c)| c.as_ref().map(|(w, _)| std::cmp::Reverse((w.clone(), i))))
            .collect();
        let mut docs: Vec<(u32, u32)> = Vec::new();
        while let Some(std::cmp::Reverse((word, first))) = heap.pop() {
            let mut from = vec![first];
            while heap.peek().is_some_and(|std::cmp::Reverse((w, _))| *w == word) {
                from.push(heap.pop().unwrap().0 .1);
            }
            from.sort_unstable();
            docs.clear();
            for &i in &from {
                docs.extend_from_slice(&current[i].as_ref().unwrap().1);
                current[i] = read_run_record(&mut readers[i])?;
                if let Some((w, _)) = &current[i] {
                    heap.push(std::cmp::Reverse((w.clone(), i)));
                }
            }
            lex.write(&word, &docs)?;
        }
        for p in &runs {
            let _ = std::fs::remove_file(p);
        }
    }
    lex.finish()?;

    let mut first = 0u64;
    let mut arts = BufWriter::new(File::create(dir.join("articles.bin"))?);
    for list in &art_docs {
        arts.write_all(&first.to_le_bytes())?;
        arts.write_all(&(list.len() as u32).to_le_bytes())?;
        first += list.len() as u64;
    }
    arts.flush()?;
    write_u32s(&dir.join("art_docs.bin"), art_docs.into_iter().flatten())?;
    hashes.par_sort_unstable();
    let mut hw = BufWriter::new(File::create(dir.join("hashes.bin"))?);
    for h in hashes {
        hw.write_all(&h.to_le_bytes())?;
    }
    hw.flush()?;
    Ok(SegMeta { dir: name, start_doc, end_doc: doc, start_byte, end_byte, total_len })
}

// ------------------------------------------------------ passages in memory

/// Passages added since the last commit, indexed in memory.
#[derive(Default)]
struct Tail {
    base: u32,
    docs: Vec<Passage>,
    hashes: HashSet<u64>,
    postings: HashMap<String, Vec<(u32, u32)>>,
    lengths: Vec<u32>,
    total_len: u64,
    articles: HashMap<String, Vec<u32>>,
    position: Vec<Option<usize>>,
}

impl Tail {
    fn index(&mut self, p: Passage) {
        let doc = self.base + self.docs.len() as u32;
        self.hashes.insert(fnv1a(normalize(&p.text).as_bytes()));
        let counts = index_terms(&p);
        let len: u32 = counts.values().sum();
        self.lengths.push(len);
        self.total_len += len as u64;
        for (term, n) in counts {
            self.postings.entry(term).or_default().push((doc, n));
        }
        self.position.push(p.article().map(|a| {
            let list = self.articles.entry(a).or_default();
            list.push(doc);
            list.len() - 1
        }));
        self.docs.push(p);
    }
}

// ------------------------------------------------------------- the store

pub struct KnowledgeStore {
    path: PathBuf,
    dir: PathBuf,
    segments: Vec<Segment>,
    tail: Tail,
    reader: Option<File>,
    /// postings per sorted run when committing (small in tests, to exercise merging)
    chunk_postings: usize,
    max_unindexed: u64,
}

/// Where a passage lives.
enum Loc<'a> {
    Seg(&'a Segment),
    Tail(usize),
}

impl KnowledgeStore {
    pub fn path_for(settings: &Settings) -> PathBuf {
        settings.data_path("knowledge_base").join("passages.jsonl")
    }

    pub fn open(settings: &Settings) -> Result<Self> {
        Self::open_at(Self::path_for(settings), CHUNK_POSTINGS, MAX_UNINDEXED_BYTES)
    }

    fn open_at(path: PathBuf, chunk_postings: usize, max_unindexed: u64) -> Result<Self> {
        let dir = path.with_file_name("index");
        let unindexed = unindexed_bytes(&path, &dir);
        if unindexed > max_unindexed {
            eprintln!("Indexing {:.0} MB of knowledge-base passages on disk (once; it can take a while)...", unindexed as f64 / 1e6);
            build_index(&path, &dir, chunk_postings, true)?;
        }
        // A merge in another process can delete a segment between reading
        // the manifest and mapping it; the new manifest is then already there.
        for _ in 0..5 {
            if let Ok(kb) = Self::try_open(&path, chunk_postings, max_unindexed, true) {
                return Ok(kb);
            }
            std::thread::sleep(std::time::Duration::from_millis(50));
        }
        // index damaged (e.g. a folder deleted by hand): read the passages
        // directly; the next commit rebuilds it
        Self::try_open(&path, chunk_postings, max_unindexed, false)
    }

    fn try_open(path: &Path, chunk_postings: usize, max_unindexed: u64, use_index: bool) -> Result<Self> {
        let dir = path.with_file_name("index");
        let file_len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        let mut manifest = read_manifest(&dir);
        if !use_index || manifest.segments.last().is_some_and(|s| s.end_byte > file_len) {
            manifest.segments.clear(); // passages.jsonl was replaced: the index is stale
        }
        let segments = manifest.segments.into_iter().map(|m| Segment::open(&dir, m)).collect::<Result<Vec<_>>>()?;
        let (base, start_byte) = segments.last().map_or((0, 0), |s| (s.meta.end_doc, s.meta.end_byte));
        let mut kb = KnowledgeStore {
            path: path.to_path_buf(),
            dir,
            segments,
            tail: Tail { base, ..Tail::default() },
            reader: File::open(path).ok(),
            chunk_postings,
            max_unindexed,
        };
        if let Some(f) = &mut kb.reader {
            f.seek(SeekFrom::Start(start_byte))?;
            let mut buf = Vec::new();
            f.read_to_end(&mut buf)?;
            let end = buf.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
            for line in buf[..end].split(|&b| b == b'\n') {
                if let Ok(p) = serde_json::from_slice::<Passage>(line) {
                    kb.tail.index(p);
                }
            }
        }
        Ok(kb)
    }

    fn contains_hash(&self, h: u64) -> bool {
        self.tail.hashes.contains(&h) || self.segments.iter().any(|s| s.contains_hash(h))
    }

    /// Store a passage. Returns its id, or None if it was already stored.
    pub fn add(&mut self, text: &str, metadata: Map<String, Value>) -> Result<Option<String>> {
        let text = normalize(text);
        let hash = fnv1a(text.as_bytes());
        if text.is_empty() || self.contains_hash(hash) {
            return Ok(None);
        }
        let p = Passage { id: format!("{hash:016x}"), text, metadata };
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let mut line = serde_json::to_string(&p)?;
        line.push('\n');
        // one write per line, so another process appending at the same time can't interleave
        std::fs::OpenOptions::new().create(true).append(true).open(&self.path)?.write_all(line.as_bytes())?;
        if self.reader.is_none() {
            self.reader = File::open(&self.path).ok();
        }
        let id = p.id.clone();
        self.tail.index(p);
        Ok(Some(id))
    }

    /// For imports of millions of passages: appends straight to disk without
    /// indexing in memory. Call `commit` afterwards to index them.
    pub fn bulk(&self) -> Result<BulkWriter<'_>> {
        if let Some(dir) = self.path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let f = std::fs::OpenOptions::new().create(true).append(true).open(&self.path)?;
        Ok(BulkWriter { kb: self, out: BufWriter::with_capacity(1 << 20, f), seen: HashSet::new(), added: 0 })
    }

    /// Index everything added since the last commit into an on-disk segment,
    /// merging small segments so there are never more than a few dozen.
    /// If another process is committing right now, does nothing (it will
    /// pick these passages up too).
    pub fn commit(&mut self) -> Result<()> {
        if build_index(&self.path, &self.dir, self.chunk_postings, false)? {
            let path = self.path.clone();
            *self = Self::open_at(path, self.chunk_postings, self.max_unindexed)?;
        }
        Ok(())
    }

    /// `commit` once enough passages have piled up in memory.
    pub fn commit_if_large(&mut self) -> Result<()> {
        if self.tail.docs.len() >= TAIL_COMMIT {
            self.commit()?;
        }
        Ok(())
    }

    pub fn count(&self) -> usize {
        self.tail.base as usize + self.tail.docs.len()
    }

    /// (on-disk segments, passages indexed on disk, passages indexed in memory, index bytes on disk)
    pub fn stats(&self) -> (usize, usize, usize, u64) {
        let bytes = self
            .segments
            .iter()
            .map(|s| [&s.lex, &s.words, &s.postings, &s.docs, &s.articles, &s.art_docs, &s.hashes].iter().map(|b| b.len() as u64).sum::<u64>())
            .sum();
        (self.segments.len(), self.tail.base as usize, self.tail.docs.len(), bytes)
    }

    fn locate(&self, doc: u32) -> Loc<'_> {
        if doc >= self.tail.base {
            return Loc::Tail((doc - self.tail.base) as usize);
        }
        let i = self.segments.partition_point(|s| s.meta.end_doc <= doc);
        Loc::Seg(&self.segments[i])
    }

    fn passage(&self, doc: u32) -> Passage {
        match self.locate(doc) {
            Loc::Tail(i) => self.tail.docs[i].clone(),
            Loc::Seg(s) => {
                let r = s.doc(doc);
                let mut buf = vec![0u8; r.line_len as usize];
                self.reader
                    .as_ref()
                    .and_then(|f| read_at(f, &mut buf, r.off).ok())
                    .and_then(|_| serde_json::from_slice(&buf).ok())
                    .unwrap_or_else(|| Passage { id: String::new(), text: String::new(), metadata: Map::new() })
            }
        }
    }

    fn text_len(&self, doc: u32) -> usize {
        match self.locate(doc) {
            Loc::Tail(i) => self.tail.docs[i].text.len(),
            Loc::Seg(s) => s.doc(doc).text_len as usize,
        }
    }

    /// Every (doc, count) for a word, in doc order, across all segments and memory.
    fn postings<'a>(&'a self, term: &str) -> impl Iterator<Item = (u32, u32)> + 'a {
        let on_disk: Vec<Postings<'a>> = self.segments.iter().filter_map(|s| s.lookup(term).map(|(_, p)| p)).collect();
        let in_mem = self.tail.postings.get(term).map(|v| v.as_slice()).unwrap_or(&[]);
        on_disk.into_iter().flatten().chain(in_mem.iter().copied())
    }

    fn df(&self, term: &str) -> usize {
        let on_disk: usize = self.segments.iter().filter_map(|s| s.lookup(term)).map(|(df, _)| df as usize).sum();
        on_disk + self.tail.postings.get(term).map_or(0, |v| v.len())
    }

    fn idf_of(&self, df: usize) -> f64 {
        let n = self.count() as f64;
        let df = df as f64;
        (1.0 + (n - df + 0.5) / (df + 0.5)).ln()
    }

    /// (doc, BM25 score, relevance) for the top-k passages with relevance
    /// (score as a fraction of what a typical passage containing every query
    /// word once would get) at least `min_relevance`.
    fn rank(&self, query: &str, k: usize, min_relevance: f64) -> Vec<(u32, f64, f64)> {
        let mut terms: Vec<String> = content_words(query);
        terms.sort();
        terms.dedup();
        let n = self.count();
        if n == 0 || terms.is_empty() || k == 0 {
            return Vec::new();
        }
        let total_len: u64 = self.segments.iter().map(|s| s.meta.total_len).sum::<u64>() + self.tail.total_len;
        let avg = total_len as f64 / n as f64;
        let avg = if avg > 0.0 { avg } else { 1.0 };
        let dfs: Vec<usize> = terms.iter().map(|t| self.df(t)).collect();
        // A word no passage contains counts as if one did - otherwise one
        // unheard-of word would swamp the rest of the query.
        let best: f64 = dfs.iter().map(|&df| self.idf_of(df.max(1))).sum();

        // Document at a time: walk every word's postings in doc order together,
        // keeping only the k best - no per-passage score table, whatever the size.
        let mut lists: Vec<(f64, std::iter::Peekable<Box<dyn Iterator<Item = (u32, u32)> + '_>>)> = terms
            .iter()
            .zip(&dfs)
            .filter(|(_, &df)| df > 0)
            .map(|(t, &df)| (self.idf_of(df), (Box::new(self.postings(t)) as Box<dyn Iterator<Item = (u32, u32)>>).peekable()))
            .collect();
        let mut top: Vec<(f64, u32)> = Vec::with_capacity(k + 1);
        loop {
            let Some(doc) = lists.iter_mut().filter_map(|(_, it)| it.peek().map(|p| p.0)).min() else { break };
            let len = match self.locate(doc) {
                Loc::Tail(i) => self.tail.lengths[i],
                Loc::Seg(s) => s.doc(doc).length,
            } as f64;
            let mut score = 0.0;
            for (idf, it) in lists.iter_mut() {
                if let Some(&(d, tf)) = it.peek() {
                    if d == doc {
                        let tf = tf as f64;
                        score += *idf * tf * (K1 + 1.0) / (tf + K1 * (1.0 - B + B * len / avg));
                        it.next();
                    }
                }
            }
            // docs arrive in increasing order, so on a tie the earlier one stays ahead
            if top.len() < k || score > top[top.len() - 1].0 {
                let at = top.partition_point(|&(s, _)| s >= score);
                top.insert(at, (score, doc));
                top.truncate(k);
            }
        }
        top.into_iter()
            .map(|(s, doc)| (doc, s, (s / best).min(1.0)))
            .filter(|&(_, _, r)| r >= min_relevance)
            .collect()
    }

    fn hit(&self, passage: Passage, score: f64, relevance: f64, via: String) -> Hit {
        Hit {
            passage,
            score: (score * 1000.0).round() / 1000.0,
            relevance: (relevance * 1000.0).round() / 1000.0,
            via,
        }
    }

    /// Top-k passages by BM25 relevance (no links followed).
    pub fn search(&self, query: &str, k: usize, min_relevance: f64) -> Vec<Hit> {
        self.rank(query, k, min_relevance)
            .into_iter()
            .map(|(doc, s, r)| self.hit(self.passage(doc), s, r, "match".into()))
            .collect()
    }

    /// Passages linked to `doc`, with link strength in (0, 1]:
    /// - neighbors in the same article: 0.8 next door, 0.4 two away (an
    ///   article's passages are stored together, so neighbors are looked up
    ///   within the segment, or the in-memory part, that holds `doc`)
    /// - passages sharing at least 2 of its 6 rarest words: up to 0.6
    fn links(&self, doc: u32, passage: &Passage) -> Vec<(u32, f64, String)> {
        let mut out: HashMap<u32, (f64, String)> = HashMap::new();
        let mut offer = |d: u32, w: f64, why: String| {
            if d != doc && out.get(&d).is_none_or(|(old, _)| w > *old) {
                out.insert(d, (w, why));
            }
        };
        let neighbors: Option<(Vec<u32>, usize)> = match self.locate(doc) {
            Loc::Tail(i) => self.tail.position[i].zip(passage.article()).map(|(pos, a)| (self.tail.articles[&a].clone(), pos)),
            Loc::Seg(s) => {
                let r = s.doc(doc);
                (r.article != NO_ARTICLE).then(|| (s.article_docs(r.article), r.pos as usize))
            }
        };
        if let (Some((list, pos)), Some(article)) = (neighbors, passage.article()) {
            for (dist, w) in [(1usize, 0.8), (2, 0.4)] {
                for p in [pos.checked_sub(dist), Some(pos + dist)].into_iter().flatten() {
                    if let Some(&d) = list.get(p) {
                        offer(d, w, format!("linked: next to it in {article}"));
                    }
                }
            }
        }
        // A word in a thousandth of everything isn't rare, and its postings are long.
        let rare_max = (self.count() / 1000).max(1000);
        let mut rare: Vec<(String, usize)> = term_counts(&passage.text)
            .into_keys()
            .map(|t| {
                let df = self.df(&t);
                (t, df)
            })
            .filter(|&(_, df)| df <= rare_max)
            .collect();
        rare.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
        rare.truncate(6);
        let mut shared: HashMap<u32, Vec<&str>> = HashMap::new();
        for (t, _) in &rare {
            for (d, _) in self.postings(t) {
                shared.entry(d).or_default().push(t.as_str());
            }
        }
        for (d, mut words) in shared {
            if words.len() >= 2 {
                words.sort();
                let w = 0.6 * words.len() as f64 / rare.len() as f64;
                offer(d, w, format!("linked: shares {}", words.join(", ")));
            }
        }
        out.into_iter().map(|(d, (w, why))| (d, w, why)).collect()
    }

    /// Like Obsidian following links from a note: the best `k` matches, then
    /// the passages they link to (strongest links first), as long as the
    /// total text fits in `budget_chars`. Best matches come first.
    pub fn search_linked(&self, query: &str, k: usize, min_relevance: f64, budget_chars: usize) -> Vec<Hit> {
        let seeds = self.rank(query, k, min_relevance);
        let mut chosen: HashSet<u32> = HashSet::new();
        let mut out = Vec::new();
        let mut used = 0usize;
        let mut neighbors: Vec<(u32, f64, String)> = Vec::new();
        for &(doc, s, r) in &seeds {
            let passage = self.passage(doc);
            for (d, w, why) in self.links(doc, &passage) {
                neighbors.push((d, r * w, why));
            }
            // The best match is always used, even if it alone overflows (the
            // prompt is cut from the left, keeping the part nearest the claim).
            if chosen.is_empty() || used + passage.text.len() <= budget_chars {
                chosen.insert(doc);
                used += passage.text.len();
                out.push(self.hit(passage, s, r, "match".into()));
            }
        }
        neighbors.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        for (d, strength, why) in neighbors {
            if strength < LINK_MIN {
                break;
            }
            let len = self.text_len(d);
            if !chosen.contains(&d) && used + len <= budget_chars {
                chosen.insert(d);
                used += len;
                out.push(self.hit(self.passage(d), 0.0, strength, why));
            }
        }
        out
    }
}

/// Bytes of passages.jsonl that no intact on-disk segment covers.
fn unindexed_bytes(path: &Path, dir: &Path) -> u64 {
    let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let m = read_manifest(dir);
    let intact = m.segments.iter().all(|s| dir.join(&s.dir).join("docs.bin").is_file());
    match m.segments.last() {
        Some(s) if intact && s.end_byte <= len => len - s.end_byte,
        _ => len,
    }
}

/// Index everything after the last on-disk segment into a new segment, and
/// merge. Returns false if another process holds the lock (unless `wait`).
fn build_index(path: &Path, dir: &Path, chunk_postings: usize, wait: bool) -> Result<bool> {
    std::fs::create_dir_all(dir)?;
    let lock = File::create(dir.join(".lock"))?;
    if wait {
        lock.lock()?;
    } else if lock.try_lock().is_err() {
        return Ok(false);
    }
    let end = complete_len(path)?;
    let mut manifest = read_manifest(dir);
    let intact = manifest.segments.iter().all(|s| dir.join(&s.dir).join("docs.bin").is_file());
    if !intact || manifest.segments.last().is_some_and(|s| s.end_byte > end) {
        manifest.segments.clear(); // damaged or stale: start over
    }
    let (start_doc, start_byte) = manifest.segments.last().map_or((0, 0), |s| (s.end_doc, s.end_byte));
    if end > start_byte {
        manifest.segments.push(build_segment(path, dir, start_doc, start_byte, end, chunk_postings)?);
        // like a binary counter: a segment at least half its predecessor's size merges into it
        while let [.., a, b] = manifest.segments.as_slice() {
            if (b.end_doc - b.start_doc) * 2 < a.end_doc - a.start_doc {
                break;
            }
            let merged = build_segment(path, dir, a.start_doc, a.start_byte, b.end_byte, chunk_postings)?;
            manifest.segments.truncate(manifest.segments.len() - 2);
            manifest.segments.push(merged);
        }
        manifest.segments.retain(|s| s.end_doc > s.start_doc || s.end_byte > s.start_byte);
        manifest.version = INDEX_VERSION;
        write_atomic(&dir.join("manifest.json"), &serde_json::to_vec_pretty(&manifest)?)?;
    }
    // remove segments no longer listed (and leftovers of interrupted builds);
    // on Windows one still mapped by a running `ask` stays until next time
    let keep: HashSet<&str> = manifest.segments.iter().map(|s| s.dir.as_str()).collect();
    for e in std::fs::read_dir(dir)?.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if name.starts_with("seg-") && !keep.contains(name.as_str()) {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
    drop(lock);
    Ok(true)
}

/// Appends passages to passages.jsonl without indexing them (see `KnowledgeStore::bulk`).
pub struct BulkWriter<'a> {
    kb: &'a KnowledgeStore,
    out: BufWriter<File>,
    seen: HashSet<u64>,
    pub added: usize,
}

impl BulkWriter<'_> {
    /// Returns false if this text is already stored (or empty).
    pub fn add(&mut self, text: &str, metadata: Map<String, Value>) -> Result<bool> {
        let text = normalize(text);
        let hash = fnv1a(text.as_bytes());
        if text.is_empty() || self.kb.contains_hash(hash) || !self.seen.insert(hash) {
            return Ok(false);
        }
        let mut line = serde_json::to_string(&Passage { id: format!("{hash:016x}"), text, metadata })?;
        line.push('\n');
        self.out.write_all(line.as_bytes())?;
        self.added += 1;
        Ok(true)
    }

    pub fn finish(mut self) -> Result<usize> {
        self.out.flush()?;
        Ok(self.added)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::testing;

    fn meta(title: &str) -> Map<String, Value> {
        let mut m = Map::new();
        m.insert("title".into(), Value::from(title));
        m
    }

    #[test]
    fn ranks_dedupes_and_persists() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let mut kb = KnowledgeStore::open(&s).unwrap();
        assert!(kb.search("anything", 3, 0.0).is_empty());
        kb.add("Usage-based pricing charges customers per unit consumed.", meta("Pricing")).unwrap();
        kb.add("Glass refracts light because light slows down in it.", meta("Optics")).unwrap();
        kb.add("Subscription pricing charges a flat monthly fee.", meta("Subscriptions")).unwrap();
        assert!(kb.add("Glass refracts  light because\nlight slows down in it.", meta("dupe")).unwrap().is_none());

        let hits = kb.search("should we move to usage-based pricing?", 2, 0.0);
        assert_eq!(hits.iter().map(|h| h.passage.meta("title")).collect::<Vec<_>>(), vec!["Pricing", "Subscriptions"]);
        let top = kb.search("refracts light glass", 5, 0.35);
        assert_eq!(top.len(), 1);
        assert!(top[0].passage.meta("title") == "Optics" && top[0].relevance > 0.35 && top[0].relevance <= 1.0);
        assert!(kb.search("kangaroo", 5, 0.0).is_empty());

        let reopened = KnowledgeStore::open(&s).unwrap();
        assert_eq!(reopened.count(), 3);
    }

    #[test]
    fn a_passage_is_found_by_its_article_title() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let mut kb = KnowledgeStore::open(&s).unwrap();
        kb.add("Its capital and largest city is the seat of government.", meta("France")).unwrap();
        kb.add("France has many rivers; the capital is not on the coast.", meta("Rivers of Europe")).unwrap();
        kb.add("Bees gather nectar.", meta("Bees")).unwrap();
        let hits = kb.search("France capital", 2, 0.0);
        assert_eq!(hits.len(), 2);
        assert!(kb.search("france", 3, 0.3).iter().any(|h| h.passage.meta("title") == "France"));
    }

    #[test]
    fn a_weak_one_word_coincidence_is_not_relevant() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let mut kb = KnowledgeStore::open(&s).unwrap();
        kb.add("Customers pay for products that are profitable to sell.", meta("Markets")).unwrap();
        kb.add("Houses in the suburbs are built from timber.", meta("Houses")).unwrap();
        assert_eq!(kb.search("Customers will pay for a profitable subscription", 3, 0.35)[0].passage.meta("title"), "Markets");
        assert!(kb.search("Kangaroos can jump over tall houses easily", 3, 0.35).is_empty());
    }

    fn article(title: &str) -> Map<String, Value> {
        let mut m = meta(title);
        m.insert("url".into(), Value::from(format!("https://x/{title}")));
        m
    }

    #[test]
    fn linked_search_follows_article_neighbors_and_shared_words() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let mut kb = KnowledgeStore::open(&s).unwrap();
        // one article, three consecutive paragraphs; only the middle one matches the query
        kb.add("Newton studied how glass bends sunlight into a spectrum of colours.", article("Opticks")).unwrap();
        kb.add("Refraction through a prism separates white light into colours.", article("Opticks")).unwrap();
        kb.add("He measured the angles carefully with brass instruments.", article("Opticks")).unwrap();
        // another article sharing two rare words with the match, but none with the query
        kb.add("A prism splits candle light too.", article("Candles")).unwrap();
        // unrelated
        kb.add("Bees gather nectar from meadow flowers in spring.", article("Bees")).unwrap();

        let plain = kb.search("refraction separates white", 3, 0.2);
        assert_eq!(plain.len(), 1);

        let linked = kb.search_linked("refraction separates white", 3, 0.2, 10_000);
        let found: Vec<(&str, &str)> = linked.iter().map(|h| (h.passage.text.split(' ').next().unwrap(), h.via.as_str())).collect();
        assert_eq!(found[0], ("Refraction", "match"));
        assert!(found.iter().any(|(w, via)| *w == "Newton" && via.contains("next to it in Opticks")), "{found:?}");
        assert!(found.iter().any(|(w, via)| *w == "He" && via.contains("next to it")), "{found:?}");
        assert!(found.iter().any(|(w, via)| *w == "A" && via.contains("shares")), "{found:?}");
        assert!(!found.iter().any(|(w, _)| *w == "Bees"));
        // linked passages rank below the match and carry a link strength
        assert!(linked[1..].iter().all(|h| h.relevance < linked[0].relevance && h.relevance > 0.0));
    }

    #[test]
    fn linked_search_respects_the_budget_but_keeps_the_best_match() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let mut kb = KnowledgeStore::open(&s).unwrap();
        kb.add("Refraction through a prism separates white light into colours.", article("Opticks")).unwrap();
        kb.add("Newton studied how glass bends sunlight into a spectrum of colours.", article("Opticks")).unwrap();
        let tight = kb.search_linked("refraction separates white", 3, 0.2, 70);
        assert_eq!(tight.len(), 1); // the neighbor doesn't fit
        let tiny = kb.search_linked("refraction separates white", 3, 0.2, 5);
        assert_eq!(tiny.len(), 1); // the best match is kept even over budget
        assert!(kb.search_linked("kangaroo", 3, 0.2, 1000).is_empty());
    }

    /// A few hundred passages over a made-up vocabulary, 5 per article.
    fn fill(kb: &mut KnowledgeStore, n: usize, seed: u64) {
        let mut rng = crate::rng::Rng::new(seed);
        for i in 0..n {
            let len = 8 + rng.below(12);
            let text: Vec<String> = (0..len).map(|_| format!("word{}", (rng.uniform().powi(2) * 200.0) as usize)).collect();
            kb.add(&text.join(" "), article(&format!("Article {seed}-{}", i / 5))).unwrap();
        }
    }

    fn answers(kb: &KnowledgeStore) -> Vec<Vec<(String, f64, f64, String)>> {
        let mut rng = crate::rng::Rng::new(7);
        let mut out = Vec::new();
        for _ in 0..40 {
            let q: Vec<String> = (0..3).map(|_| format!("word{}", rng.below(200))).collect();
            let q = q.join(" ");
            for hits in [kb.search(&q, 5, 0.0), kb.search_linked(&q, 3, 0.1, 600)] {
                out.push(hits.into_iter().map(|h| (h.passage.id, h.score, h.relevance, h.via)).collect());
            }
        }
        out
    }

    #[test]
    fn the_on_disk_index_answers_exactly_like_the_in_memory_one() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let path = KnowledgeStore::path_for(&s);
        // tiny runs, so building the segment spills and merges many of them
        let mut kb = KnowledgeStore::open_at(path.clone(), 50, MAX_UNINDEXED_BYTES).unwrap();
        fill(&mut kb, 400, 1);
        let before = answers(&kb);
        assert!(before.iter().filter(|h| !h.is_empty()).count() > 40);
        assert!(before.iter().flatten().any(|h| h.3.contains("next to it")) && before.iter().flatten().any(|h| h.3.contains("shares")));
        kb.commit().unwrap();
        assert_eq!(kb.stats().0, 1);
        assert_eq!((kb.stats().1, kb.stats().2), (400, 0));
        assert_eq!(answers(&kb), before);
        let reopened = KnowledgeStore::open(&s).unwrap();
        assert_eq!(reopened.count(), 400);
        assert_eq!(answers(&reopened), before);
    }

    #[test]
    fn commits_merge_segments_and_still_dedupe() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let path = KnowledgeStore::path_for(&s);
        let mut kb = KnowledgeStore::open_at(path.clone(), 1000, MAX_UNINDEXED_BYTES).unwrap();
        for round in 0..9 {
            fill(&mut kb, 30, 100 + round);
            kb.commit().unwrap();
            assert!(kb.stats().0 <= 4, "{} segments after round {round}", kb.stats().0);
        }
        fill(&mut kb, 17, 555); // some left in memory
        let n = kb.count();
        assert!(n > 250 && kb.stats().2 > 0);
        let first = kb.passage(0).text;
        assert!(kb.add(&first, Map::new()).unwrap().is_none());
        // a fresh copy of the same passages without any index answers the same
        let other = tempfile::tempdir().unwrap();
        let s2 = testing::settings(other.path());
        std::fs::create_dir_all(KnowledgeStore::path_for(&s2).parent().unwrap()).unwrap();
        std::fs::copy(&path, KnowledgeStore::path_for(&s2)).unwrap();
        let plain = KnowledgeStore::open(&s2).unwrap();
        assert_eq!(plain.stats().0, 0);
        assert_eq!(plain.count(), n);
        assert_eq!(answers(&kb), answers(&plain));
        // the leftover segment folders were cleaned up
        let dirs = std::fs::read_dir(path.with_file_name("index")).unwrap().flatten()
            .filter(|e| e.file_name().to_string_lossy().starts_with("seg-")).count();
        assert_eq!(dirs, kb.stats().0);
    }

    #[test]
    fn opening_indexes_a_big_unindexed_store_instead_of_loading_it() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let path = KnowledgeStore::path_for(&s);
        let mut kb = KnowledgeStore::open(&s).unwrap();
        fill(&mut kb, 200, 4);
        let before = answers(&kb);
        // over the limit: opening builds the on-disk index first
        let big = KnowledgeStore::open_at(path.clone(), 1000, 2000).unwrap();
        assert_eq!((big.stats().1, big.stats().2), (200, 0));
        assert_eq!(answers(&big), before);
        // a segment folder deleted by hand: the index is rebuilt from scratch
        let seg = std::fs::read_dir(path.with_file_name("index")).unwrap().flatten()
            .find(|e| e.file_name().to_string_lossy().starts_with("seg-")).unwrap();
        std::fs::remove_dir_all(seg.path()).unwrap();
        let rebuilt = KnowledgeStore::open_at(path.clone(), 1000, 2000).unwrap();
        assert_eq!((rebuilt.stats().1, rebuilt.stats().2), (200, 0));
        assert_eq!(answers(&rebuilt), before);
    }

    #[test]
    fn bulk_import_then_commit() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let mut kb = KnowledgeStore::open(&s).unwrap();
        kb.add("Refraction through a prism separates white light into colours.", article("Opticks")).unwrap();
        kb.commit().unwrap();
        let mut w = kb.bulk().unwrap();
        assert!(w.add("Bees gather nectar from meadow flowers.", article("Bees")).unwrap());
        assert!(!w.add("Bees  gather nectar from meadow flowers.", article("Bees")).unwrap());
        assert!(!w.add("Refraction through a prism separates white light into colours.", Map::new()).unwrap());
        assert!(w.add("Honey is made from that nectar in the hive.", article("Bees")).unwrap());
        assert_eq!(w.finish().unwrap(), 2);
        kb.commit().unwrap();
        assert_eq!(kb.count(), 3);
        let hits = kb.search_linked("bees nectar meadow", 1, 0.2, 10_000);
        assert_eq!(hits.len(), 2);
        assert!(hits[1].via.contains("next to it in Bees"));
    }

    #[test]
    fn half_written_lines_and_stale_or_damaged_indexes_are_handled() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let path = KnowledgeStore::path_for(&s);
        let mut kb = KnowledgeStore::open(&s).unwrap();
        fill(&mut kb, 20, 3);
        kb.commit().unwrap();
        // another process is halfway through writing a line
        let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        f.write_all(br#"{"id": "x", "text": "Half a li"#).unwrap();
        assert_eq!(KnowledgeStore::open(&s).unwrap().count(), 20);
        let mut kb = KnowledgeStore::open(&s).unwrap();
        kb.commit().unwrap(); // must not index the half line
        f.write_all(b"ne about kangaroos.\"}\n").unwrap();
        let kb = KnowledgeStore::open(&s).unwrap();
        assert_eq!(kb.count(), 21);
        assert_eq!(kb.search("kangaroos", 1, 0.0)[0].passage.text, "Half a line about kangaroos.");
        // a segment folder deleted by hand: fall back to reading the passages
        let seg = std::fs::read_dir(path.with_file_name("index")).unwrap().flatten()
            .find(|e| e.file_name().to_string_lossy().starts_with("seg-")).unwrap();
        std::fs::remove_dir_all(seg.path()).unwrap();
        assert_eq!(KnowledgeStore::open(&s).unwrap().count(), 21);
        // passages.jsonl replaced with a shorter one: the index is ignored
        std::fs::write(&path, "{\"id\": \"a\", \"text\": \"Only one passage now.\"}\n").unwrap();
        let kb = KnowledgeStore::open(&s).unwrap();
        assert_eq!(kb.count(), 1);
        assert_eq!(kb.search("passage", 3, 0.0).len(), 1);
    }

    #[test]
    fn reads_passages_written_by_the_python_version() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let line = r#"{"id": "f5b7487aa8459be9", "text": "Customers pay for products.", "metadata": {"title": "Markets", "url": "https://x"}}"#;
        std::fs::write(KnowledgeStore::path_for(&s), format!("{line}\n")).unwrap();
        let mut kb = KnowledgeStore::open(&s).unwrap();
        assert_eq!(kb.count(), 1);
        assert!(kb.add("Customers pay for products.", Map::new()).unwrap().is_none());
    }
}
