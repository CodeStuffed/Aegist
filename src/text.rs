//! Word-level helpers shared by the router and the knowledge store.

pub const STOPWORDS: &[&str] = &[
    "a", "about", "above", "after", "again", "against", "all", "also", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
    "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing",
    "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how",
    "i", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "just", "let's", "may", "me",
    "might", "more", "most", "much", "must", "mustn't", "my", "myself", "no", "nor", "not", "now", "of",
    "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shall", "she", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the",
    "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd",
    "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "upon", "us", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't",
    "what", "what's", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won't",
    "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves", "one", "two", "get", "got",
    "make", "made", "really", "thing", "things", "way", "ways", "lot", "lots", "every", "going",
];

pub fn is_stopword(w: &str) -> bool {
    STOPWORDS.contains(&w)
}

/// Lowercase words (letters and digits, plus an apostrophe suffix like "don't").
pub fn words(text: &str) -> Vec<String> {
    let lower = text.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let mut out = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        if !chars[i].is_alphanumeric() {
            i += 1;
            continue;
        }
        let mut w = String::new();
        while i < chars.len() && chars[i].is_alphanumeric() {
            w.push(chars[i]);
            i += 1;
        }
        if i + 1 < chars.len() && chars[i] == '\'' && chars[i + 1].is_alphabetic() {
            w.push('\'');
            i += 1;
            while i < chars.len() && chars[i].is_alphabetic() {
                w.push(chars[i]);
                i += 1;
            }
        }
        out.push(w);
    }
    out
}

/// Words minus stopwords and very short tokens, in order.
pub fn content_words(text: &str) -> Vec<String> {
    words(text).into_iter().filter(|w| !is_stopword(w) && w.chars().count() > 2).collect()
}

/// The claim's most distinctive words: longest content words first, deduped.
pub fn keywords(text: &str, limit: usize) -> Vec<String> {
    let mut seen = std::collections::HashSet::new();
    let mut unique: Vec<String> = content_words(text)
        .into_iter()
        .filter(|w| w.chars().count() >= 4 && !w.chars().all(|c| c.is_ascii_digit()))
        .filter(|w| seen.insert(w.clone()))
        .collect();
    unique.sort_by_key(|w| std::cmp::Reverse(w.chars().count())); // stable: ties keep order
    unique.truncate(limit);
    unique
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitting() {
        assert_eq!(words("Don't PANIC: it's 42!"), vec!["don't", "panic", "it's", "42"]);
        assert_eq!(content_words("The Great Wall is visible from orbit"), vec!["great", "wall", "visible", "orbit"]);
        assert_eq!(keywords("We should switch our SaaS to usage-based pricing", 4), vec!["pricing", "switch", "usage", "based"]);
    }
}
