//! Tokenizer speed on a text file:
//!     cargo run --release --example tok_speed -- <file> <train MB> <vocab>
use council::tokenizer::Tokenizer;
use rayon::prelude::*;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).unwrap();
    let mb: usize = args.get(2).and_then(|a| a.parse().ok()).unwrap_or(3);
    let vocab: usize = args.get(3).and_then(|a| a.parse().ok()).unwrap_or(8192);
    let mut end = (mb << 20).min(text.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let t = Instant::now();
    let tok = Tokenizer::train(&text[..end], vocab);
    println!("train on {} MB -> {} tokens in {:.1}s", end >> 20, tok.vocab_size(), t.elapsed().as_secs_f64());
    let chunks: Vec<&str> = text.split("\n\n").collect();
    let t = Instant::now();
    let n: usize = chunks.par_iter().map(|c| tok.encode(c).len()).sum();
    let s = t.elapsed().as_secs_f64();
    println!("encode {:.0} MB -> {n} tokens in {s:.1}s: {:.1} MB/s on {} threads", text.len() as f64 / 1e6, text.len() as f64 / 1e6 / s, rayon::current_num_threads());
}
