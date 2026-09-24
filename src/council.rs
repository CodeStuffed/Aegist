//! The council: Believer and Skeptic always, Investor for money ideas, then
//! the Judge - plus familiarity caps and the repeat-run consistency check.
//! `evaluate` is the entry point the CLI (and any other program) calls.
//!
//! A persona (config/personas/<name>.yaml) is a lead-in the model continues
//! to write its position, plus probe phrases it is scored on. Its signal is
//! pointwise mutual information: how much more likely the model finds the
//! probes right after the claim than after a neutral lead-in (nats/token).

use crate::brain::Brain;
use crate::config::{load_persona, Persona, Settings};
use crate::knowledge::{Hit, KnowledgeStore};
use crate::rng::Rng;
use crate::router::{route, Routing};
use crate::session_log::{self, Session};
use crate::util::iso_now;
use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Confidence {
    Low,
    Medium,
    High,
}

impl std::fmt::Display for Confidence {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Stance {
    Yes,
    No,
    Mixed,
}

impl std::fmt::Display for Stance {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str(match self { Stance::Yes => "yes", Stance::No => "no", Stance::Mixed => "mixed" })
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct Evidence {
    pub delta: f32,
    pub text: String,
    pub source: String,
    pub url: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct PersonaResult {
    pub persona: String,
    pub role: String,
    pub position: String,
    pub key_points: Vec<Evidence>,
    pub signal: f32,
    pub confidence: Confidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_before_cap: Option<Confidence>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Judgement {
    pub verdict: String,
    pub stance: Stance,
    pub margin: f32,
    pub reasoning: String,
    pub in_its_own_words: String,
    pub relied_on: Vec<String>,
    pub confidence: Confidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_before_cap: Option<Confidence>,
    pub unresolved: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Familiarity {
    pub claim_loss: f32,
    pub typical_loss: f32,
    pub ratio: f32,
    pub cap: Confidence,
    pub notes: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelInfo {
    pub description: String,
    pub steps: u64,
    pub tokens_seen: u64,
    pub val_loss: Option<f32>,
    pub val_bpb: Option<f32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Consistency {
    pub agreed: bool,
    pub reasons: Vec<String>,
    pub rerun_judge: Judgement,
}

#[derive(Clone, Debug, Serialize)]
pub struct Evaluation {
    pub claim: String,
    pub timestamp: String,
    pub model: ModelInfo,
    pub routing: Routing,
    pub memory: Vec<Hit>,
    pub familiarity: Familiarity,
    pub panel: Vec<PersonaResult>,
    pub judge: Judgement,
    pub overall_confidence: Confidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consistency: Option<Consistency>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_note: Option<String>,
}

pub const LOW_NOTE: &str = "Low confidence: the council didn't agree with itself on a repeat run";

pub fn confidence_from_signal(signal: f32, settings: &Settings) -> Confidence {
    let t = &settings.council.confidence_thresholds;
    if signal >= t.high {
        Confidence::High
    } else if signal >= t.medium {
        Confidence::Medium
    } else {
        Confidence::Low
    }
}

pub fn as_sentence(claim: &str) -> String {
    let c = claim.split_whitespace().collect::<Vec<_>>().join(" ");
    if c.ends_with(['.', '!', '?']) { c } else { format!("{c}.") }
}

/// Evidence first, claim last. Hits arrive most relevant first; they go in
/// reverse so the best sits next to the claim, since a long prompt is cut
/// from the left to fit the model's context.
pub fn build_prompt(claim: &str, passages: &[&Hit]) -> String {
    let mut parts: Vec<String> = passages.iter().rev().map(|h| h.passage.text.clone()).collect();
    parts.push(as_sentence(claim));
    parts.join("\n\n")
}

/// How much knowledge-base text fits in front of the claim: the model's
/// context minus room for the claim and the generated position, at roughly
/// 3.5 characters per token.
pub fn context_budget_chars(brain: &Brain, settings: &Settings) -> usize {
    let reserve = settings.council.generate.max_new_tokens + 32;
    (brain.model.cfg.block_size.saturating_sub(reserve) as f64 * 3.5) as usize
}

pub fn assess_familiarity(brain: &Brain, claim: &str, settings: &Settings) -> Familiarity {
    let cfg = &settings.council;
    let typical = brain.meta.stats.val_loss.unwrap_or(f32::NAN);
    let claim_loss = brain.text_nll(claim);
    let ratio = if typical.is_finite() && typical > 0.0 { claim_loss / typical } else { f32::INFINITY };
    let mut cap = if ratio > cfg.familiarity.low_above {
        Confidence::Low
    } else if ratio > cfg.familiarity.medium_above {
        Confidence::Medium
    } else {
        Confidence::High
    };
    let mut notes = Vec::new();
    if cap != Confidence::High {
        notes.push(format!("the claim is unlike most of what the model has read (loss {claim_loss:.2} vs. a typical {typical:.2})"));
    }
    let seen = brain.meta.stats.tokens_seen;
    if seen < cfg.min_tokens_trained {
        cap = Confidence::Low;
        notes.push(format!("the model has only trained on {:.1}M tokens (under {:.0}M)", seen as f64 / 1e6, cfg.min_tokens_trained as f64 / 1e6));
    }
    Familiarity { claim_loss, typical_loss: typical, ratio, cap, notes }
}

/// Scores every probe of every persona after one prompt, sharing one pass
/// over the prompt. PMI = score after prompt - score after neutral lead-in.
struct Scorer<'a> {
    brain: &'a Brain,
    probes: Vec<String>,
    index: HashMap<String, usize>,
    baseline: Vec<f32>,
}

impl<'a> Scorer<'a> {
    fn new(brain: &'a Brain, personas: &[Persona], settings: &Settings) -> Self {
        let mut probes: Vec<String> = Vec::new();
        for p in personas {
            for q in p.probes.iter().chain(&p.counter_probes) {
                if !probes.contains(q) {
                    probes.push(q.clone());
                }
            }
        }
        let index = probes.iter().enumerate().map(|(i, q)| (q.clone(), i)).collect();
        let baseline = if probes.is_empty() { Vec::new() } else { brain.score(&settings.council.neutral_prompt, &probes, None) };
        Scorer { brain, probes, index, baseline }
    }

    fn pmi_all(&self, prompt: &str, rng: Option<&mut Rng>) -> Vec<f32> {
        if self.probes.is_empty() {
            return Vec::new();
        }
        let scores = self.brain.score(prompt, &self.probes, rng);
        scores.iter().zip(&self.baseline).map(|(s, b)| s - b).collect()
    }

    fn signal(&self, pmi: &[f32], persona: &Persona) -> f32 {
        let mean = |qs: &[String]| {
            if qs.is_empty() {
                0.0
            } else {
                qs.iter().map(|q| pmi[self.index[q]]).sum::<f32>() / qs.len() as f32
            }
        };
        mean(&persona.probes) - mean(&persona.counter_probes)
    }
}

pub struct Pass {
    pub panel: Vec<PersonaResult>,
    pub judge: Judgement,
}

fn speak(brain: &Brain, prompt: &str, persona: &Persona, settings: &Settings, rng: &mut Rng) -> String {
    let g = &settings.council.generate;
    let text = brain.generate(&format!("{prompt} {}", persona.lead_in), g.max_new_tokens, g.temperature, g.top_k, rng);
    format!("{} {text}", persona.lead_in).trim().to_string()
}

/// One pass: panel then Judge. `stochastic` switches dropout on while
/// scoring (Monte Carlo dropout), used by the repeat run.
pub fn run_council(
    brain: &Brain, claim: &str, routing: &Routing, passages: &[Hit], familiarity: &Familiarity,
    settings: &Settings, stochastic: bool, seed: u64,
) -> Result<Pass> {
    let mut names = vec!["believer", "skeptic"];
    if routing.is_money_idea {
        names.push("investor");
    }
    let personas: Vec<Persona> = names.iter().map(|n| load_persona(settings, n)).collect::<Result<_>>()?;
    let judge_persona = load_persona(settings, "judge")?;
    let scorer = Scorer::new(brain, &personas, settings);
    let mut score_rng = Rng::new(seed ^ 0xD20);
    let mut gen_rng = Rng::new(seed ^ 0x6E4);
    let rng_opt = |r: &mut Rng| if stochastic { Some(Rng::new(r.next_u64())) } else { None };

    let refs: Vec<&Hit> = passages.iter().collect();
    let full_prompt = build_prompt(claim, &refs);
    let full = scorer.pmi_all(&full_prompt, rng_opt(&mut score_rng).as_mut());
    let bare = if passages.is_empty() { full.clone() } else { scorer.pmi_all(&build_prompt(claim, &[]), rng_opt(&mut score_rng).as_mut()) };
    let per_passage: Vec<Vec<f32>> = refs.iter().map(|h| scorer.pmi_all(&build_prompt(claim, &[*h]), rng_opt(&mut score_rng).as_mut())).collect();

    let mut panel = Vec::new();
    for (name, persona) in names.iter().zip(&personas) {
        let signal = scorer.signal(&full, persona);
        let without = scorer.signal(&bare, persona);
        // Which stored passages push the model toward this persona's side?
        let mut key_points: Vec<Evidence> = passages
            .iter()
            .zip(&per_passage)
            .filter_map(|(hit, pmi)| {
                let delta = scorer.signal(pmi, persona) - without;
                (delta > 0.0).then(|| Evidence {
                    delta: (delta * 1000.0).round() / 1000.0,
                    text: hit.passage.text.clone(),
                    source: [hit.passage.meta("title"), hit.passage.meta("source")].into_iter().find(|s| !s.is_empty()).unwrap_or("").to_string(),
                    url: hit.passage.meta("url").to_string(),
                })
            })
            .collect();
        key_points.sort_by(|a, b| b.delta.total_cmp(&a.delta));
        let own = confidence_from_signal(signal, settings);
        let confidence = own.min(familiarity.cap);
        panel.push(PersonaResult {
            persona: name.to_string(),
            role: persona.role.clone(),
            position: speak(brain, &full_prompt, persona, settings, &mut gen_rng),
            key_points,
            signal: (signal * 1000.0).round() / 1000.0,
            confidence,
            confidence_before_cap: (confidence != own).then_some(own),
        });
    }
    let own_words = speak(brain, &full_prompt, &judge_persona, settings, &mut gen_rng);
    let judge = judge(&panel, familiarity, settings, own_words);
    Ok(Pass { panel, judge })
}

/// The Judge: stance from the margin between the sides, confidence never
/// above the weakest input relied on nor above what familiarity allows.
pub fn judge(panel: &[PersonaResult], familiarity: &Familiarity, settings: &Settings, own_words: String) -> Judgement {
    let get = |n: &str| panel.iter().find(|p| p.persona == n);
    let (believer, skeptic) = (get("believer").expect("believer"), get("skeptic").expect("skeptic"));
    let investor = get("investor");
    let truth_margin = believer.signal - skeptic.signal;
    let margin = match investor { Some(i) => (truth_margin + i.signal) / 2.0, None => truth_margin };
    let mm = settings.council.mixed_margin;
    let (stance, mut relied_on) = if margin > mm {
        (Stance::Yes, vec!["believer".to_string()])
    } else if margin < -mm {
        (Stance::No, vec!["skeptic".to_string()])
    } else {
        (Stance::Mixed, panel.iter().map(|p| p.persona.clone()).collect())
    };
    if let Some(i) = investor {
        if stance != Stance::Mixed && (i.signal > 0.0) == (stance == Stance::Yes) {
            relied_on.push("investor".into());
        }
    }
    let own = confidence_from_signal(margin.abs(), settings);
    let confidence = relied_on
        .iter()
        .filter_map(|n| get(n).map(|p| p.confidence))
        .fold(own.min(familiarity.cap), Confidence::min);

    let mut reasoning = format!(
        "Believer signal {:+.2} vs. Skeptic {:+.2} nats/token (margin {truth_margin:+.2}).",
        believer.signal, skeptic.signal
    );
    if let Some(i) = investor {
        reasoning += &format!(" Investor {:+.2}; combined margin {margin:+.2}.", i.signal);
    }
    let mut unresolved = Vec::new();
    if stance == Stance::Mixed {
        unresolved.push("neither side's signal clearly beats the other".to_string());
    }
    unresolved.extend(familiarity.notes.iter().cloned());
    let verdict = match stance {
        Stance::Yes => "Yes - on what this model has read, the claim holds up.",
        Stance::No => "No - on what this model has read, the claim doesn't hold up.",
        Stance::Mixed => "Undecided - the model finds the claim about as compatible with 'true' as with 'false'.",
    };
    Judgement {
        verdict: verdict.to_string(),
        stance,
        margin: (margin * 1000.0).round() / 1000.0,
        reasoning,
        in_its_own_words: own_words,
        relied_on,
        confidence,
        confidence_before_cap: (confidence != own).then_some(own),
        unresolved: (!unresolved.is_empty()).then(|| unresolved.join("; ")),
    }
}

/// Material disagreements between two verdicts (empty if none).
pub fn compare_verdicts(a: &Judgement, b: &Judgement) -> Vec<String> {
    let mut reasons = Vec::new();
    if a.stance != b.stance {
        reasons.push(format!("verdict flipped: {} -> {}", a.stance, b.stance));
    }
    if a.confidence != b.confidence {
        reasons.push(format!("confidence changed: {} -> {}", a.confidence, b.confidence));
    }
    reasons
}

pub struct Options {
    /// Re-run with dropout on and compare; None = uncertainty.enabled.
    pub recheck: Option<bool>,
    /// Log the session so the research loop learns what you ask about.
    pub record: bool,
    pub seed: Option<u64>,
}

impl Default for Options {
    fn default() -> Self {
        Options { recheck: None, record: true, seed: None }
    }
}

/// Run one full council session on `claim`.
pub fn evaluate(brain: &Brain, claim: &str, settings: &Settings, opts: &Options, log: &mut dyn FnMut(&str)) -> Result<Evaluation> {
    let routing = route(claim, settings);
    let kb = KnowledgeStore::open(settings)?;
    let mem = &settings.memory;
    let memory = if mem.follow_links {
        kb.search_linked(claim, mem.top_k, mem.min_relevance, context_budget_chars(brain, settings))
    } else {
        kb.search(claim, mem.top_k, mem.min_relevance)
    };
    let linked = memory.iter().filter(|h| h.via != "match").count();
    log(&format!("Found {} relevant passage(s) in the knowledge base ({linked} by following links)", memory.len()));
    let familiarity = assess_familiarity(brain, claim, settings);
    let seed = opts.seed.unwrap_or_else(|| Rng::from_time().next_u64());

    log(&format!("Council deliberating: {}", if routing.is_money_idea { "Believer, Skeptic, Investor, Judge" } else { "Believer, Skeptic, Judge" }));
    let first = run_council(brain, claim, &routing, &memory, &familiarity, settings, false, seed)?;
    let mut overall = first.judge.confidence;
    let (mut consistency, mut note) = (None, None);
    if opts.recheck.unwrap_or(settings.uncertainty.enabled) {
        log("Re-running with dropout on, to check the council agrees with itself");
        let rerun = run_council(brain, claim, &routing, &memory, &familiarity, settings, true, seed.wrapping_add(1))?;
        let reasons = compare_verdicts(&first.judge, &rerun.judge);
        if !reasons.is_empty() {
            overall = Confidence::Low;
            note = Some(LOW_NOTE.to_string());
        }
        consistency = Some(Consistency { agreed: reasons.is_empty(), reasons, rerun_judge: rerun.judge });
    }
    let st = &brain.meta.stats;
    let result = Evaluation {
        claim: claim.to_string(),
        timestamp: iso_now(),
        model: ModelInfo { description: brain.describe(), steps: st.steps, tokens_seen: st.tokens_seen, val_loss: st.val_loss, val_bpb: st.val_bpb },
        routing,
        memory,
        familiarity,
        panel: first.panel,
        judge: first.judge,
        overall_confidence: overall,
        consistency,
        confidence_note: note,
    };
    if opts.record {
        session_log::record(settings, &Session {
            at: result.timestamp.clone(),
            claim: result.claim.clone(),
            keywords: result.routing.keywords.clone(),
            stance: result.judge.stance.to_string(),
            confidence: result.overall_confidence.to_string(),
        })?;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trainer::tests::trained;

    fn member(name: &str, signal: f32, confidence: Confidence) -> PersonaResult {
        PersonaResult { persona: name.into(), role: String::new(), position: String::new(), key_points: vec![], signal, confidence, confidence_before_cap: None }
    }

    fn familiar() -> Familiarity {
        Familiarity { claim_loss: 3.0, typical_loss: 3.0, ratio: 1.0, cap: Confidence::High, notes: vec![] }
    }

    fn settings() -> (tempfile::TempDir, Settings) {
        let tmp = tempfile::tempdir().unwrap();
        let s = crate::config::testing::settings(tmp.path());
        (tmp, s)
    }

    #[test]
    fn judge_can_say_no_and_undecided() {
        let (_t, s) = settings();
        let no = judge(&[member("believer", 0.1, Confidence::High), member("skeptic", 0.9, Confidence::High)], &familiar(), &s, String::new());
        assert_eq!((no.stance, no.confidence, no.relied_on.clone()), (Stance::No, Confidence::High, vec!["skeptic".to_string()]));
        assert!(no.verdict.starts_with("No"));
        let mixed = judge(&[member("believer", 0.30, Confidence::High), member("skeptic", 0.25, Confidence::High)], &familiar(), &s, String::new());
        assert_eq!((mixed.stance, mixed.confidence), (Stance::Mixed, Confidence::Low));
        assert!(mixed.unresolved.is_some());
    }

    #[test]
    fn judge_is_capped_by_weakest_input_and_familiarity() {
        let (_t, s) = settings();
        let capped = judge(&[member("believer", 0.9, Confidence::Medium), member("skeptic", 0.0, Confidence::High)], &familiar(), &s, String::new());
        assert_eq!((capped.stance, capped.confidence, capped.confidence_before_cap), (Stance::Yes, Confidence::Medium, Some(Confidence::High)));
        let unfamiliar = Familiarity { cap: Confidence::Low, notes: vec!["the claim is unlike most of what it read".into()], ..familiar() };
        let low = judge(&[member("believer", 0.9, Confidence::High), member("skeptic", 0.0, Confidence::High)], &unfamiliar, &s, String::new());
        assert!(low.confidence == Confidence::Low && low.unresolved.unwrap().contains("unlike"));
    }

    #[test]
    fn investor_moves_the_margin() {
        let (_t, s) = settings();
        let panel = |inv: f32| vec![member("believer", 0.5, Confidence::High), member("skeptic", 0.3, Confidence::High), member("investor", inv, Confidence::High)];
        assert_eq!(judge(&panel(-1.0), &familiar(), &s, String::new()).stance, Stance::No);
        let yes = judge(&panel(0.8), &familiar(), &s, String::new());
        assert_eq!((yes.stance, yes.relied_on), (Stance::Yes, vec!["believer".to_string(), "investor".to_string()]));
    }

    #[test]
    fn consistency_check() {
        let (_t, s) = settings();
        let a = judge(&[member("believer", 0.9, Confidence::High), member("skeptic", 0.0, Confidence::High)], &familiar(), &s, String::new());
        let mut b = a.clone();
        assert!(compare_verdicts(&a, &b).is_empty());
        b.stance = Stance::No;
        b.confidence = Confidence::Medium;
        assert_eq!(compare_verdicts(&a, &b), vec!["verdict flipped: yes -> no".to_string(), "confidence changed: High -> Medium".to_string()]);
    }

    #[test]
    fn build_prompt_puts_best_passage_next_to_claim() {
        let (_t, s) = settings();
        let mut kb = KnowledgeStore::open(&s).unwrap();
        kb.add("best passage about light", Default::default()).unwrap();
        kb.add("second passage about light glass", Default::default()).unwrap();
        let hits = kb.search("light", 2, 0.0);
        let prompt = build_prompt("Claim", &hits.iter().collect::<Vec<_>>());
        assert!(prompt.ends_with(&format!("{}\n\nClaim.", hits[0].passage.text)));
    }

    #[test]
    fn evaluate_end_to_end() {
        let (_tmp, s) = trained(40);
        let brain = Brain::load(&s).unwrap();
        let mut kb = KnowledgeStore::open(&s).unwrap();
        let mut meta = serde_json::Map::new();
        meta.insert("title".into(), "Markets".into());
        kb.add("Customers pay for products that are profitable to sell.", meta).unwrap();

        let r = evaluate(&brain, "Customers will pay for a profitable subscription", &s, &Options { seed: Some(3), ..Default::default() }, &mut |_| {}).unwrap();
        assert_eq!(r.panel.iter().map(|p| p.persona.as_str()).collect::<Vec<_>>(), vec!["believer", "skeptic", "investor"]);
        assert_eq!(r.memory[0].passage.meta("title"), "Markets");
        assert!(r.panel[0].position.starts_with("This is true because"));
        assert!(r.consistency.is_some());
        assert_eq!(session_log::sessions(&s).last().unwrap().claim, "Customers will pay for a profitable subscription");
        let json = serde_json::to_value(&r).unwrap();
        assert!(json["judge"]["stance"].is_string() && json["overall_confidence"].is_string());

        let plain = evaluate(&brain, "Light bends in glass", &s, &Options { recheck: Some(false), record: false, seed: Some(4) }, &mut |_| {}).unwrap();
        assert!(plain.panel.len() == 2 && plain.consistency.is_none());
        assert_eq!(session_log::sessions(&s).len(), 1);
    }
}
