//! Autonomous research + self-training loop. Until time runs out:
//!   1. pick a topic: seed_topics from settings, then words that keep coming
//!      up in your past council sessions (a frequency count)
//!   2. fetch Wikipedia articles on it (plain search API - no AI involved)
//!   3. store their passages in the knowledge base and the full text in the
//!      training corpus
//!   4. keep training the model on the grown corpus until the next fetch
//! Fetches are capped per hour (politeness to Wikipedia); training fills the
//! time in between. State survives restarts in data/research_state.json.

use crate::config::Settings;
use crate::corpus::corpus_dir;
use crate::knowledge::KnowledgeStore;
use crate::session_log;
use crate::trainer::NotEnoughText;
use crate::util::{iso_now, write_atomic};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};

const HOUR: f64 = 3600.0;
const USER_AGENT: &str = "council-engine/0.2 (self-hosted research loop; personal use)";
const STOP_SECTIONS: &[&str] = &["references", "see also", "external links", "notes", "further reading", "bibliography", "sources", "citations"];

#[derive(Clone, Debug)]
pub struct Article {
    pub title: String,
    pub url: String,
    pub text: String,
}

#[derive(Default, Serialize, Deserialize)]
pub struct State {
    #[serde(default)]
    pub fetches: Vec<f64>,
    #[serde(default)]
    pub topics: BTreeMap<String, f64>,
}

pub fn load_state(settings: &Settings) -> State {
    std::fs::read(settings.data_path("").join("research_state.json"))
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

fn save_state(state: &State, settings: &Settings) -> Result<()> {
    write_atomic(&settings.data_path("").join("research_state.json"), &serde_json::to_vec_pretty(state)?)
}

/// 0 if under max_topics_per_hour for the trailing hour, else the wait.
pub fn seconds_until_fetch_allowed(state: &mut State, settings: &Settings, now: f64) -> f64 {
    state.fetches.retain(|&t| now - t < HOUR);
    state.fetches.sort_by(f64::total_cmp);
    let cap = settings.research.max_topics_per_hour;
    if state.fetches.len() < cap {
        0.0
    } else {
        state.fetches[state.fetches.len() - cap] + HOUR - now
    }
}

/// Seed topics first, then recurring session keywords (most frequent first),
/// minus anything fetched within refresh_hours.
pub fn due_topics(state: &State, settings: &Settings, now: f64) -> Vec<String> {
    let fresh = settings.research.refresh_hours * HOUR;
    let mut out: Vec<String> = Vec::new();
    let seeds = settings.research.seed_topics.iter().map(|t| t.trim().to_lowercase());
    let frequent = session_log::keyword_counts(settings).into_iter().map(|(k, _)| k);
    for t in seeds.chain(frequent) {
        if !out.contains(&t) && state.topics.get(&t).map_or(true, |&at| now - at >= fresh) {
            out.push(t);
        }
    }
    out
}

pub fn fetch_wikipedia(topic: &str, settings: &Settings) -> Result<Vec<Article>> {
    let cfg = &settings.research;
    let mut builder = ureq::AgentBuilder::new().timeout(std::time::Duration::from_secs(30)).user_agent(USER_AGENT);
    if let Some(proxy) = proxy_for(&cfg.wikipedia_api) {
        builder = builder.proxy(ureq::Proxy::new(proxy)?);
    }
    let agent = builder.build();
    let found: Value = agent
        .get(&cfg.wikipedia_api)
        .query("action", "query")
        .query("list", "search")
        .query("srsearch", topic)
        .query("srlimit", &cfg.articles_per_topic.to_string())
        .query("format", "json")
        .query("formatversion", "2")
        .call()?
        .into_json()?;
    let mut articles = Vec::new();
    for hit in found["query"]["search"].as_array().into_iter().flatten() {
        let Some(title) = hit["title"].as_str() else { continue };
        let page: Value = agent
            .get(&cfg.wikipedia_api)
            .query("action", "query")
            .query("prop", "extracts")
            .query("explaintext", "1")
            .query("redirects", "1")
            .query("titles", title)
            .query("format", "json")
            .query("formatversion", "2")
            .call()?
            .into_json()?;
        for p in page["query"]["pages"].as_array().into_iter().flatten() {
            if let (Some(t), Some(text)) = (p["title"].as_str(), p["extract"].as_str()) {
                if text.is_empty() {
                    continue;
                }
                let base = cfg.wikipedia_api.replace("/w/api.php", "/wiki/");
                let url = format!("{base}{}", url_encode(&t.replace(' ', "_")));
                articles.push(Article { title: t.to_string(), url, text: text.to_string() });
            }
        }
    }
    Ok(articles)
}

/// The proxy to use for `url` from HTTPS_PROXY / HTTP_PROXY, honoring
/// NO_PROXY (which the HTTP library's own env support ignores).
fn proxy_for(url: &str) -> Option<String> {
    let env = |names: &[&str]| names.iter().find_map(|n| std::env::var(n).ok().filter(|v| !v.is_empty()));
    let (scheme, rest) = url.split_once("://")?;
    let authority = rest.split(['/', '?']).next()?;
    let host = match authority.rsplit_once(':') {
        Some((h, port)) if port.chars().all(|c| c.is_ascii_digit()) => h,
        _ => authority,
    };
    let host = host.trim_matches(['[', ']']).to_lowercase();
    if matches!(host.as_str(), "localhost" | "127.0.0.1" | "::1") {
        return None;
    }
    let no_proxy = env(&["NO_PROXY", "no_proxy"]).unwrap_or_default();
    for rule in no_proxy.split(',').map(|r| r.trim().to_lowercase()).filter(|r| !r.is_empty()) {
        let rule = rule.trim_start_matches("*.").trim_start_matches('.').to_string();
        if rule == "*" || host == rule || host.ends_with(&format!(".{rule}")) {
            return None;
        }
    }
    match scheme {
        "https" => env(&["HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"]),
        _ => env(&["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"]),
    }
}

fn url_encode(s: &str) -> String {
    s.bytes()
        .map(|b| match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' | b'(' | b')' => (b as char).to_string(),
            _ => format!("%{b:02X}"),
        })
        .collect()
}

/// Article body -> paragraphs, minus headings and reference sections.
pub fn split_passages(text: &str, min_chars: usize) -> Vec<String> {
    let mut out = Vec::new();
    for line in text.lines() {
        let t = line.trim();
        if t.starts_with("==") && t.ends_with("==") {
            let heading = t.trim_matches('=').trim().to_lowercase();
            if STOP_SECTIONS.contains(&heading.as_str()) {
                break;
            }
            continue;
        }
        if t.chars().count() >= min_chars {
            out.push(t.split_whitespace().collect::<Vec<_>>().join(" "));
        }
    }
    out
}

fn slug(text: &str) -> String {
    let s: String = text.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
    let s: String = s.split('-').filter(|p| !p.is_empty()).collect::<Vec<_>>().join("-");
    let s: String = s.chars().take(80).collect();
    if s.is_empty() { "article".into() } else { s }
}

/// Passages -> knowledge base; full text -> training corpus. Returns how many
/// new passages were stored.
pub fn store_articles(topic: &str, articles: &[Article], settings: &Settings, kb: &mut KnowledgeStore) -> Result<usize> {
    let corpus = corpus_dir(settings).join("research");
    std::fs::create_dir_all(&corpus)?;
    let retrieved_at = iso_now();
    let mut stored = 0;
    for a in articles {
        let passages = split_passages(&a.text, settings.research.min_passage_chars);
        for p in &passages {
            let mut meta = Map::new();
            for (k, v) in [("topic", topic), ("title", &a.title), ("url", &a.url), ("retrieved_at", &retrieved_at), ("source", "wikipedia")] {
                meta.insert(k.into(), Value::from(v));
            }
            if kb.add(p, meta)?.is_some() {
                stored += 1;
            }
        }
        if !passages.is_empty() {
            std::fs::write(corpus.join(format!("{}.txt", slug(&a.title))), passages.join("\n\n"))?;
        }
    }
    kb.commit_if_large()?;
    Ok(stored)
}

pub trait Clock {
    fn now(&self) -> f64;
    fn sleep(&self, seconds: f64);
}

pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> f64 {
        crate::util::unix_now()
    }
    fn sleep(&self, seconds: f64) {
        std::thread::sleep(std::time::Duration::from_secs_f64(seconds.max(0.0)));
    }
}

#[derive(Debug, Default, PartialEq)]
pub struct Summary {
    pub topics: Vec<String>,
    pub passages_stored: usize,
    pub training_minutes: f64,
    pub interrupted: bool,
}

/// Research and self-train for `hours`. `fetch` and `train` are passed in so
/// tests can swap them; `train` returns Ok(true) if it was interrupted.
pub fn run(
    hours: f64,
    settings: &Settings,
    log: &mut dyn FnMut(String),
    clock: &dyn Clock,
    fetch: &mut dyn FnMut(&str) -> Result<Vec<Article>>,
    train: &mut dyn FnMut(f64, &mut dyn FnMut(String)) -> Result<bool>,
    stop: &AtomicBool,
) -> Result<Summary> {
    let deadline = clock.now() + hours * HOUR;
    let mut state = load_state(settings);
    let mut kb = KnowledgeStore::open(settings)?;
    let mut summary = Summary::default();
    let burst_s = settings.research.train_minutes_per_topic * 60.0;

    while clock.now() < deadline {
        if stop.load(Ordering::Relaxed) {
            summary.interrupted = true;
            break;
        }
        let now = clock.now();
        let topics = due_topics(&state, settings, now);
        if !topics.is_empty() && seconds_until_fetch_allowed(&mut state, settings, now) == 0.0 {
            let topic = topics[0].clone();
            state.fetches.push(now);
            state.topics.insert(topic.clone(), now);
            save_state(&state, settings)?;
            match fetch(&topic) {
                Ok(articles) => {
                    let stored = store_articles(&topic, &articles, settings, &mut kb)?;
                    let titles = if articles.is_empty() { "nothing found".to_string() } else { articles.iter().map(|a| a.title.as_str()).collect::<Vec<_>>().join(", ") };
                    log(format!("Researched '{topic}': {stored} new passages ({titles})."));
                    summary.topics.push(topic);
                    summary.passages_stored += stored;
                }
                Err(e) => {
                    // Still counts toward the hourly cap, but the topic stays due.
                    state.topics.remove(&topic);
                    save_state(&state, settings)?;
                    log(format!("Couldn't fetch '{topic}' from Wikipedia: {e}"));
                }
            }
        }

        // Self-train until the next fetch is due.
        let burst = burst_s.min(deadline - clock.now());
        if burst <= 0.0 {
            break;
        }
        let started = clock.now();
        match train(burst / 60.0, log) {
            Ok(interrupted) => {
                summary.training_minutes += (clock.now() - started) / 60.0;
                if interrupted {
                    summary.interrupted = true;
                    break;
                }
            }
            Err(e) if e.downcast_ref::<NotEnoughText>().is_some() => {
                log(format!("Not training yet: {e}"));
                // Nothing to train on: wait for the next fetch slot instead.
                let now = clock.now();
                let idle = if due_topics(&state, settings, now).is_empty() { burst } else { seconds_until_fetch_allowed(&mut state, settings, now) };
                clock.sleep(idle.max(1.0).min(burst).min((deadline - now).max(0.0)));
            }
            Err(e) => return Err(e),
        }
    }
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::testing;
    use std::cell::Cell;
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    const ARTICLE: &str = "Pricing is the process of setting a price. Pricing is the process of setting a price. Pricing is the process of setting a price. Pricing is the process.\n\
Value-based pricing sets prices by what customers will pay. Value-based pricing sets prices by what customers will pay. Value-based pricing sets prices.\n\
== History ==\n\
Short line.\n\
Early merchants priced goods by haggling in open markets over many centuries. Early merchants priced goods by haggling in open markets over many centuries.\n\
== References ==\n\
Smith, A. (1776). The Wealth of Nations. A very long citation line that should be dropped because it is in the references.";

    struct FakeClock(Cell<f64>);
    impl Clock for FakeClock {
        fn now(&self) -> f64 { self.0.get() }
        fn sleep(&self, s: f64) { self.0.set(self.0.get() + s) }
    }

    /// A tiny HTTP server that answers like Wikipedia's API; returns its base URL.
    fn fake_wikipedia() -> (String, std::sync::Arc<std::sync::Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let mut reader = BufReader::new(stream.try_clone().unwrap());
                let mut first = String::new();
                reader.read_line(&mut first).unwrap();
                let mut line = String::new();
                while reader.read_line(&mut line).map(|n| n > 2).unwrap_or(false) { line.clear(); }
                let path = first.split_whitespace().nth(1).unwrap_or("").to_string();
                log.lock().unwrap().push(path.clone());
                let body = if path.contains("list=search") {
                    r#"{"query":{"search":[{"title":"Pricing"},{"title":"Price discrimination"}]}}"#.to_string()
                } else {
                    let title = if path.contains("titles=Pricing") { "Pricing" } else { "Price discrimination" };
                    serde_json::json!({"query": {"pages": [{"title": title, "extract": ARTICLE.replace("Pricing", title)}]}}).to_string()
                };
                let mut s = stream;
                let _ = write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", body.len());
            }
        });
        (format!("http://{addr}/w/api.php"), seen)
    }

    #[test]
    fn local_addresses_never_use_a_proxy() {
        assert_eq!(proxy_for("http://127.0.0.1:8080/w/api.php"), None);
        assert_eq!(proxy_for("http://localhost/w/api.php"), None);
    }

    #[test]
    fn split_passages_drops_headings_short_lines_and_references() {
        let p = split_passages(ARTICLE, 100);
        assert_eq!(p.len(), 3);
        assert!(!p.iter().any(|x| x.contains("==") || x.contains("Wealth of Nations") || x == "Short line."));
    }

    #[test]
    fn fetches_from_the_plain_search_api_and_stores() {
        let tmp = tempfile::tempdir().unwrap();
        let mut s = testing::settings(tmp.path());
        let (api, seen) = fake_wikipedia();
        s.research.wikipedia_api = api;
        s.research.min_passage_chars = 100;
        let articles = fetch_wikipedia("pricing strategy", &s).unwrap();
        assert_eq!(articles.iter().map(|a| a.title.as_str()).collect::<Vec<_>>(), vec!["Pricing", "Price discrimination"]);
        assert!(articles[1].url.ends_with("/wiki/Price_discrimination"));
        let first = seen.lock().unwrap()[0].clone();
        assert!(first.contains("srsearch=pricing+strategy") || first.contains("srsearch=pricing%20strategy"));
        assert!(first.contains("srlimit=2"));

        let mut kb = KnowledgeStore::open(&s).unwrap();
        // 3 passages from the first article; the second repeats two of them word for word
        assert_eq!(store_articles("pricing", &articles, &s, &mut kb).unwrap(), 4);
        assert_eq!(store_articles("pricing", &articles, &s, &mut kb).unwrap(), 0);
        let hit = &kb.search("value-based pricing customers", 1, 0.0)[0];
        assert!(hit.passage.meta("url").starts_with("http") && hit.passage.meta("topic") == "pricing");
        assert_eq!(std::fs::read_dir(corpus_dir(&s).join("research")).unwrap().count(), 2);
    }

    #[test]
    fn due_topics_seeds_then_frequent_session_words() {
        let tmp = tempfile::tempdir().unwrap();
        let mut s = testing::settings(tmp.path());
        s.research.seed_topics = vec!["Pricing Strategy".into()];
        for kw in [vec!["churn"], vec!["churn", "onboarding"]] {
            session_log::record(&s, &session_log::Session { at: "t".into(), claim: "c".into(), keywords: kw.into_iter().map(String::from).collect(), stance: "yes".into(), confidence: "Low".into() }).unwrap();
        }
        let mut state = State::default();
        assert_eq!(due_topics(&state, &s, 0.0), vec!["pricing strategy", "churn", "onboarding"]);
        state.topics.insert("churn".into(), 0.0);
        assert!(!due_topics(&state, &s, HOUR).contains(&"churn".to_string()));
        assert!(due_topics(&state, &s, 169.0 * HOUR).contains(&"churn".to_string()));
    }

    #[test]
    fn rate_limit_per_trailing_hour() {
        let tmp = tempfile::tempdir().unwrap();
        let mut s = testing::settings(tmp.path());
        s.research.max_topics_per_hour = 2;
        let mut state = State { fetches: vec![0.0, 600.0], topics: BTreeMap::new() };
        assert_eq!(seconds_until_fetch_allowed(&mut state, &s, 1200.0), 2400.0);
        assert_eq!(seconds_until_fetch_allowed(&mut state, &s, 3601.0), 0.0);
    }

    #[test]
    fn alternates_research_and_training_and_survives_restart() {
        let tmp = tempfile::tempdir().unwrap();
        let mut s = testing::settings(tmp.path());
        s.research.seed_topics = vec!["pricing".into(), "markets".into(), "profit".into()];
        s.research.max_topics_per_hour = 2;
        s.research.train_minutes_per_topic = 5.0;
        s.research.min_passage_chars = 100;
        let clock = FakeClock(Cell::new(1e6));
        let trained = Cell::new(0.0);
        let mut fetch = |t: &str| Ok(vec![Article { title: t.to_string(), url: format!("https://x/{t}"), text: ARTICLE.replace("Pricing", t) }]);
        let mut train = |minutes: f64, _: &mut dyn FnMut(String)| { trained.set(trained.get() + minutes); clock.0.set(clock.0.get() + minutes * 60.0); Ok(false) };
        let summary = run(0.5, &s, &mut |_| {}, &clock, &mut fetch, &mut train, &AtomicBool::new(false)).unwrap();
        // 30 minutes: fetch, train 5, fetch, train 5, then the hourly cap blocks fetches
        assert_eq!(summary.topics, vec!["pricing", "markets"]);
        assert!((trained.get() - 30.0).abs() < 1e-6);
        assert!(KnowledgeStore::open(&s).unwrap().count() > 0);
        let state = load_state(&s);
        assert_eq!((state.topics.len(), state.fetches.len()), (2, 2));
    }

    #[test]
    fn failed_fetch_keeps_topic_due_and_waits_when_nothing_to_train() {
        let tmp = tempfile::tempdir().unwrap();
        let mut s = testing::settings(tmp.path());
        s.research.seed_topics = vec!["pricing".into()];
        s.research.max_topics_per_hour = 3;
        let clock = FakeClock(Cell::new(1e6));
        let mut fetch = |_: &str| -> Result<Vec<Article>> { anyhow::bail!("no network") };
        let mut train = |_: f64, _: &mut dyn FnMut(String)| -> Result<bool> { Err(NotEnoughText("empty".into()).into()) };
        let summary = run(0.1, &s, &mut |_| {}, &clock, &mut fetch, &mut train, &AtomicBool::new(false)).unwrap();
        assert!(summary.topics.is_empty() && summary.training_minutes == 0.0);
        let state = load_state(&s);
        assert!(state.topics.is_empty() && state.fetches.len() == 3); // retried, but only up to the cap
        assert!((clock.now() - (1e6 + 0.1 * HOUR)).abs() < 1.0);
    }

    #[test]
    fn stop_flag_ends_the_loop() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let clock = FakeClock(Cell::new(0.0));
        let summary = run(1.0, &s, &mut |_| {}, &clock, &mut |_| Ok(vec![]), &mut |_, _| Ok(false), &AtomicBool::new(true)).unwrap();
        assert!(summary.interrupted);
    }
}
