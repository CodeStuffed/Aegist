//! Byte-level BPE tokenizer, trained from scratch on the corpus.
//!
//! Text is split into word-ish chunks, each chunk becomes its UTF-8 bytes
//! (ids 0-255, so any text at all can be encoded), and training repeatedly
//! merges the most frequent adjacent pair into a new token.

use std::collections::{HashMap, HashSet};

pub const BYTE_VOCAB: u32 = 256;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Class {
    Letter,
    Digit,
    Space,
    Other,
}

fn class(c: char) -> Class {
    if c.is_alphabetic() {
        Class::Letter
    } else if c.is_ascii_digit() {
        Class::Digit
    } else if c.is_whitespace() {
        Class::Space
    } else {
        Class::Other
    }
}

/// Split text into chunks: an optional leading space plus a run of letters
/// (any script),
/// up to three digits, or other symbols; or a run of whitespace. A run of
/// whitespace leaves its last space for the word after it, so " word" is
/// always one chunk no matter how much space came before.
pub fn pretokenize(text: &str) -> Vec<&str> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        let start = chars[i].0;
        let mut j = i;
        if chars[j].1 == ' ' && j + 1 < chars.len() && class(chars[j + 1].1) != Class::Space {
            j += 1; // the space belongs to the next chunk
        }
        let kind = class(chars[j].1);
        match kind {
            Class::Space => {
                // whitespace run; stop before a final ' ' that precedes a word
                while j < chars.len() && class(chars[j].1) == Class::Space {
                    if chars[j].1 == ' '
                        && j + 1 < chars.len()
                        && class(chars[j + 1].1) != Class::Space
                        && j > i
                    {
                        break;
                    }
                    j += 1;
                }
            }
            Class::Digit => {
                let mut n = 0;
                while j < chars.len() && class(chars[j].1) == Class::Digit && n < 3 {
                    j += 1;
                    n += 1;
                }
            }
            _ => {
                while j < chars.len() && class(chars[j].1) == kind {
                    j += 1;
                }
            }
        }
        let end = if j < chars.len() { chars[j].0 } else { text.len() };
        out.push(&text[start..end]);
        i = j;
    }
    out
}

#[derive(Clone, Debug)]
pub struct Tokenizer {
    pub merges: Vec<(u32, u32)>,
    ranks: HashMap<(u32, u32), u32>,
    vocab: Vec<Vec<u8>>,
}

impl Tokenizer {
    pub fn new(merges: Vec<(u32, u32)>) -> Self {
        let ranks = merges.iter().enumerate().map(|(i, &p)| (p, i as u32)).collect();
        let mut vocab: Vec<Vec<u8>> = (0..BYTE_VOCAB).map(|b| vec![b as u8]).collect();
        for &(a, b) in &merges {
            let mut bytes = vocab[a as usize].clone();
            bytes.extend_from_slice(&vocab[b as usize]);
            vocab.push(bytes);
        }
        Tokenizer { merges, ranks, vocab }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// How many bytes of text a token stands for.
    pub fn token_len(&self, id: u32) -> usize {
        self.vocab.get(id as usize).map_or(0, |v| v.len())
    }

    /// Learn merges until the vocabulary reaches `vocab_size` or nothing repeats.
    pub fn train(text: &str, vocab_size: usize) -> Self {
        let mut freqs: HashMap<&str, i64> = HashMap::new();
        for chunk in pretokenize(text) {
            *freqs.entry(chunk).or_default() += 1;
        }
        let mut words: Vec<Vec<u32>> = Vec::with_capacity(freqs.len());
        let mut counts: Vec<i64> = Vec::with_capacity(freqs.len());
        let mut sorted: Vec<_> = freqs.into_iter().collect();
        sorted.sort_unstable(); // deterministic order
        for (w, c) in sorted {
            words.push(w.bytes().map(u32::from).collect());
            counts.push(c);
        }

        let mut pair_counts: HashMap<(u32, u32), i64> = HashMap::new();
        let mut where_: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();
        for (wi, w) in words.iter().enumerate() {
            for p in w.windows(2) {
                let pair = (p[0], p[1]);
                *pair_counts.entry(pair).or_default() += counts[wi];
                where_.entry(pair).or_default().insert(wi);
            }
        }

        let mut merges = Vec::new();
        while (BYTE_VOCAB as usize + merges.len()) < vocab_size {
            let best = pair_counts
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
                .map(|(&p, &c)| (p, c));
            let Some((best, count)) = best else { break };
            if count < 2 {
                break;
            }
            let new_id = BYTE_VOCAB + merges.len() as u32;
            merges.push(best);
            let affected = where_.remove(&best).unwrap_or_default();
            for wi in affected {
                let c = counts[wi];
                let old = std::mem::take(&mut words[wi]);
                for p in old.windows(2) {
                    let pair = (p[0], p[1]);
                    if let Some(v) = pair_counts.get_mut(&pair) {
                        *v -= c;
                        if *v <= 0 {
                            pair_counts.remove(&pair);
                        }
                    }
                }
                let mut merged = Vec::with_capacity(old.len());
                let mut i = 0;
                while i < old.len() {
                    if i + 1 < old.len() && (old[i], old[i + 1]) == best {
                        merged.push(new_id);
                        i += 2;
                    } else {
                        merged.push(old[i]);
                        i += 1;
                    }
                }
                for p in merged.windows(2) {
                    let pair = (p[0], p[1]);
                    *pair_counts.entry(pair).or_default() += c;
                    where_.entry(pair).or_default().insert(wi);
                }
                words[wi] = merged;
            }
            pair_counts.remove(&best);
        }
        Tokenizer::new(merges)
    }

    fn encode_chunk(&self, chunk: &str, out: &mut Vec<u32>) {
        let mut seq: Vec<u32> = chunk.bytes().map(u32::from).collect();
        while seq.len() > 1 {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..seq.len() - 1 {
                if let Some(&r) = self.ranks.get(&(seq[i], seq[i + 1])) {
                    if best.map_or(true, |(br, _)| r < br) {
                        best = Some((r, i));
                    }
                }
            }
            let Some((rank, i)) = best else { break };
            seq[i] = BYTE_VOCAB + rank;
            seq.remove(i + 1);
        }
        out.extend_from_slice(&seq);
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::with_capacity(text.len() / 3);
        let mut cache: HashMap<&str, (usize, usize)> = HashMap::new();
        for chunk in pretokenize(text) {
            if let Some(&(start, len)) = cache.get(chunk) {
                out.extend_from_within(start..start + len);
                continue;
            }
            let start = out.len();
            self.encode_chunk(chunk, &mut out);
            cache.insert(chunk, (start, out.len() - start));
        }
        out
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&i| self.vocab.get(i as usize).map(|v| v.as_slice()).unwrap_or(&[]))
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT: &str = "The rays of light are refracted by the prism. The prism separates the colours. ";

    #[test]
    fn pretokenize_keeps_every_byte_and_attaches_spaces_to_words() {
        let s = "Hello  world,  it's 12345 — 日本 ✓\n\n  ok";
        assert_eq!(pretokenize(s).concat(), s);
        let chunks = pretokenize("a  word");
        assert_eq!(chunks, vec!["a", " ", " word"]);
        assert_eq!(pretokenize("12345"), vec!["123", "45"]);
    }

    #[test]
    fn roundtrip_any_unicode() {
        let tok = Tokenizer::train(&TEXT.repeat(30), 300);
        for s in [TEXT, "Héllo wörld — 日本語 ✓ $9.99/month", "", "\n\n  tabs\there"] {
            assert_eq!(tok.decode(&tok.encode(s)), s);
        }
    }

    #[test]
    fn training_compresses_and_respects_vocab_size() {
        let text = TEXT.repeat(30);
        let tok = Tokenizer::train(&text, 300);
        assert_eq!(tok.vocab_size(), 300);
        assert!(tok.encode(&text).len() < text.len() / 3);
        assert!(Tokenizer::train("abc", 1000).vocab_size() < 1000);
        let again = Tokenizer::new(tok.merges.clone());
        assert_eq!(again.encode(&text), tok.encode(&text));
    }
}
