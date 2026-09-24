//! The council: Believer and Skeptic always, Investor for money ideas, then
//! the Judge - plus familiarity caps and the repeat-run consistency check.
//! `evaluate` is the entry point the CLI (and any other program) calls.
//!
//! A persona (config/personas/<name>.yaml) is a lead-in the model continues
//! to write its position, plus probe phrases it is scored on. Its signal is
//! pointwise mutual information: how much more likely the model finds the
//! probes right after the claim than after a neutral lead-in (nats/token).
//!
//! The negation test asks the model more directly: with the evidence in
//! front, does the rest of the claim read as more likely after "X is" or
//! after "X is not"? A model that has read a lot of text knows how things
//! are usually said, and this puts that knowledge to work.

use crate::brain::{Brain, Prefix};
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

/// The claim with its main verb's polarity flipped ("X is Y" <-> "X is not Y").
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Negation {
    /// the claim up to and including the flipped verb, as stated and flipped
    pub stated_prefix: String,
    pub flipped_prefix: String,
    /// the rest of the claim, scored after each prefix
    pub rest: String,
    /// the flipped claim in full
    pub flipped: String,
    /// the same verb, as stated and flipped, after a generic subject
    /// ("It is" / "It is not"): the baseline that removes the general cost
    /// of "not" from the comparison
    pub generic_stated: String,
    pub generic_flipped: String,
}

/// How much more likely the rest of the claim is after its stated polarity
/// than after the flipped one (mean log-probability per token, in nats).
#[derive(Clone, Debug, Serialize)]
pub struct NegationTest {
    pub flipped: String,
    pub stated_logp: f32,
    pub flipped_logp: f32,
    /// > 0 supports the claim as stated: raw_contrast minus baseline
    pub contrast: f32,
    /// stated minus flipped, for the claim's own subject
    pub raw_contrast: f32,
    /// the same for a generic subject: how much "not" costs this sentence anyway
    pub baseline: f32,
    pub confidence: Confidence,
}

const AUXILIARIES: &[&str] = &["is", "are", "was", "were", "am", "will", "would", "can", "could", "should", "must", "may", "might", "shall", "does", "do", "did"];
const PERFECT: &[&str] = &["has", "have", "had"];
const CONTRACTIONS: &[(&str, &str)] = &[
    ("isn't", "is"), ("aren't", "are"), ("wasn't", "was"), ("weren't", "were"), ("won't", "will"), ("wouldn't", "would"),
    ("can't", "can"), ("cannot", "can"), ("couldn't", "could"), ("shouldn't", "should"), ("mustn't", "must"), ("doesn't", "does"),
    ("don't", "do"), ("didn't", "did"), ("hasn't", "has"), ("haven't", "have"), ("hadn't", "had"), ("mightn't", "might"),
];

/// "has/have/had" followed by one of these is the main verb ("has a dog",
/// "have to go"), not an auxiliary ("has grown").
fn starts_noun_phrase(word: &str) -> bool {
    const STARTERS: &[&str] = &[
        "a", "an", "the", "no", "not", "some", "any", "many", "much", "more", "less", "few", "several", "enough", "every", "each", "all",
        "my", "your", "his", "her", "its", "our", "their", "this", "that", "these", "those", "to", "one", "two", "three", "lots", "plenty",
    ];
    let w = word.to_lowercase();
    STARTERS.contains(&w.trim_end_matches(|c: char| !c.is_alphanumeric())) || w.starts_with(|c: char| c.is_ascii_digit())
}

/// Flip the first auxiliary verb after the subject: "Prices will rise" ->
/// "Prices will not rise", "X isn't Y" -> "X is Y". None when the claim has
/// no such verb (e.g. "Light bends in water") or is a question.
pub fn negate(claim: &str) -> Option<Negation> {
    let sentence = as_sentence(claim).replace('’', "'");
    let words: Vec<&str> = sentence.split_whitespace().collect();
    let join = |ws: &[&str]| ws.join(" ");
    for i in 1..words.len().saturating_sub(1) {
        let w = words[i];
        let lower = w.to_lowercase();
        if !lower.chars().all(|c| c.is_alphabetic() || c == '\'') {
            continue; // punctuation attached: not a verb in the middle of a clause
        }
        let (stated, flipped, rest_from) = if let Some(&(_, base)) = CONTRACTIONS.iter().find(|(c, _)| *c == lower) {
            (join(&words[..=i]), format!("{} {base}", join(&words[..i])), i + 1)
        } else if AUXILIARIES.contains(&lower.as_str())
            || (PERFECT.contains(&lower.as_str()) && !starts_noun_phrase(words[i + 1]))
        {
            if words[i + 1].eq_ignore_ascii_case("not") {
                (join(&words[..=i + 1]), join(&words[..=i]), i + 2)
            } else {
                (join(&words[..=i]), format!("{} not", join(&words[..=i])), i + 1)
            }
        } else {
            continue;
        };
        let rest = join(&words[rest_from..]);
        if !rest.chars().any(|c| c.is_alphabetic()) {
            return None;
        }
        // the verb part alone, after a subject it agrees with
        let subject = join(&words[..i]);
        let verb = |prefix: &str| prefix[subject.len()..].trim().to_string();
        let (sv, fv) = (verb(&stated), verb(&flipped));
        let base = if lower == "am" {
            "I"
        } else if ["are", "were", "have", "do", "aren't", "weren't", "haven't", "don't"].contains(&lower.as_str()) {
            "They"
        } else {
            "It"
        };
        return Some(Negation {
            flipped: format!("{flipped} {rest}"),
            stated_prefix: stated,
            flipped_prefix: flipped,
            rest,
            generic_stated: format!("{base} {sv}"),
            generic_flipped: format!("{base} {fv}"),
        });
    }
    None
}

/// Evidence, then `text` - as build_prompt, without making `text` a sentence.
pub fn evidence_then(text: &str, passages: &[&Hit]) -> String {
    let mut parts: Vec<String> = passages.iter().rev().map(|h| h.passage.text.clone()).collect();
    parts.push(text.to_string());
    parts.join("\n\n")
}

/// Room the negation test needs after the evidence.
fn negation_room(brain: &Brain, neg: &Negation) -> usize {
    let len = |t: &str| brain.tokenizer.encode(t).len();
    len(&neg.stated_prefix).max(len(&neg.flipped_prefix)) + len(&neg.rest) + 3
}

/// `evidence`: the knowledge-base passages already run through the model
/// (`evidence_prefix`), shared by both versions of the claim.
pub fn negation_test(brain: &Brain, neg: &Negation, evidence: &Prefix, settings: &Settings, mut rng: Option<&mut Rng>) -> NegationTest {
    let rest = vec![format!(" {}", neg.rest)];
    let mut after = |prefix: &str| {
        let p = brain.extend(evidence, prefix, rng.as_deref_mut());
        brain.score_after(&p, &rest, rng.as_deref_mut())[0]
    };
    let (stated, flipped) = (after(&neg.stated_prefix), after(&neg.flipped_prefix));
    let baseline = after(&neg.generic_stated) - after(&neg.generic_flipped);
    let raw = stated - flipped;
    let contrast = raw - baseline;
    let r = |x: f32| (x * 1000.0).round() / 1000.0;
    NegationTest {
        flipped: neg.flipped.clone(),
        stated_logp: r(stated),
        flipped_logp: r(flipped),
        contrast: r(contrast),
        raw_contrast: r(raw),
        baseline: r(baseline),
        confidence: confidence_from_signal(contrast.abs(), settings),
    }
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub negation_test: Option<NegationTest>,
    pub overall_confidence: Confidence,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub consistency: Option<Consistency>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confidence_note: Option<String>,
}

/// Passages checked one by one for how they moved each persona.
const MAX_ATTRIBUTED: usize = 5;

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

    fn pmi_after(&self, prefix: &Prefix, rng: Option<&mut Rng>) -> Vec<f32> {
        if self.probes.is_empty() {
            return Vec::new();
        }
        let scores = self.brain.score_after(prefix, &self.probes, rng);
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
    pub negation: Option<NegationTest>,
    pub judge: Judgement,
}

/// The persona's position, in the model's words, continuing after the prompt.
fn speak(brain: &Brain, prompt: &Prefix, persona: &Persona, settings: &Settings, rng: &mut Rng) -> String {
    let g = &settings.council.generate;
    let lead = brain.extend(prompt, &format!(" {}", persona.lead_in), None);
    let text = brain.generate_after(&lead, g.max_new_tokens, g.temperature, g.top_k, rng);
    format!("{} {text}", persona.lead_in).trim().to_string()
}

/// One pass: panel then Judge. `stochastic` switches dropout on while
/// scoring (Monte Carlo dropout), used by the repeat run - which only
/// scores: the positions in words are left empty.
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
    // The evidence runs through the model once: the claim follows it (then
    // every persona is scored against, and continues, that), and so do both
    // versions of the negation test.
    let negation = negate(claim);
    let claim_text = as_sentence(claim);
    let reserve = (brain.tokenizer.encode(&claim_text).len() + settings.council.generate.max_new_tokens + 32)
        .max(negation.as_ref().map_or(0, |n| negation_room(brain, n)));
    let evidence = brain.prefix(&evidence_then("", &refs), reserve, rng_opt(&mut score_rng).as_mut());
    let full_prefix = brain.extend(&evidence, &claim_text, rng_opt(&mut score_rng).as_mut());
    let full = scorer.pmi_after(&full_prefix, rng_opt(&mut score_rng).as_mut());
    // Which passages moved each persona (shown for the first pass only)
    let bare = if passages.is_empty() || stochastic { full.clone() } else { scorer.pmi_all(&build_prompt(claim, &[]), None) };
    let per_passage: Vec<Vec<f32>> = if stochastic {
        Vec::new()
    } else {
        // the strongest few (matches first, then the strongest links): one model pass each
        refs.iter().take(MAX_ATTRIBUTED).map(|h| scorer.pmi_all(&build_prompt(claim, &[*h]), None)).collect()
    };

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
            position: if stochastic { String::new() } else { speak(brain, &full_prefix, persona, settings, &mut gen_rng) },
            key_points,
            signal: (signal * 1000.0).round() / 1000.0,
            confidence,
            confidence_before_cap: (confidence != own).then_some(own),
        });
    }
    let negation = negation.map(|n| negation_test(brain, &n, &evidence, settings, rng_opt(&mut score_rng).as_mut()));
    let own_words = if stochastic { String::new() } else { speak(brain, &full_prefix, &judge_persona, settings, &mut gen_rng) };
    let judge = judge(&panel, negation.as_ref(), familiarity, settings, own_words);
    Ok(Pass { panel, negation, judge })
}

/// The Judge: stance from the margin between the sides (blended with the
/// negation test when the claim has one), confidence never above the
/// weakest input relied on nor above what familiarity allows.
pub fn judge(panel: &[PersonaResult], negation: Option<&NegationTest>, familiarity: &Familiarity, settings: &Settings, own_words: String) -> Judgement {
    let get = |n: &str| panel.iter().find(|p| p.persona == n);
    let (believer, skeptic) = (get("believer").expect("believer"), get("skeptic").expect("skeptic"));
    let investor = get("investor");
    let truth_margin = believer.signal - skeptic.signal;
    let panel_margin = match investor { Some(i) => (truth_margin + i.signal) / 2.0, None => truth_margin };
    let w = settings.council.negation_weight.clamp(0.0, 1.0);
    let margin = match negation { Some(n) => (1.0 - w) * panel_margin + w * n.contrast, None => panel_margin };
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
    let negation_agrees = negation.filter(|n| stance != Stance::Mixed && w > 0.0 && (n.contrast > 0.0) == (stance == Stance::Yes));
    if negation_agrees.is_some() {
        relied_on.push("negation test".into());
    }
    let own = confidence_from_signal(margin.abs(), settings);
    let confidence = relied_on
        .iter()
        .filter_map(|n| get(n).map(|p| p.confidence))
        .chain(negation_agrees.map(|n| n.confidence))
        .fold(own.min(familiarity.cap), Confidence::min);

    let mut reasoning = format!(
        "Believer signal {:+.2} vs. Skeptic {:+.2} nats/token (margin {truth_margin:+.2}).",
        believer.signal, skeptic.signal
    );
    if let Some(i) = investor {
        reasoning += &format!(" Investor {:+.2}; panel margin {panel_margin:+.2}.", i.signal);
    }
    if let Some(n) = negation {
        reasoning += &format!(" Negation test {:+.2} (the claim as stated vs. \"{}\"); combined margin {margin:+.2}.", n.contrast, n.flipped);
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
        let mut rerun = run_council(brain, claim, &routing, &memory, &familiarity, settings, true, seed.wrapping_add(1))?;
        rerun.judge.in_its_own_words = first.judge.in_its_own_words.clone();
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
        negation_test: first.negation,
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
    fn negations() {
        let n = |c: &str| negate(c).map(|n| (n.stated_prefix, n.flipped_prefix, n.rest));
        let t = |a: &str, b: &str, c: &str| Some((a.to_string(), b.to_string(), c.to_string()));
        assert_eq!(n("Light is refracted in glass"), t("Light is", "Light is not", "refracted in glass."));
        assert_eq!(n("Raising prices will reduce demand."), t("Raising prices will", "Raising prices will not", "reduce demand."));
        assert_eq!(n("Heavy objects do not fall faster"), t("Heavy objects do not", "Heavy objects do", "fall faster."));
        assert_eq!(n("We can't win this market"), t("We can't", "We can", "win this market."));
        assert_eq!(n("The firm has grown quickly"), t("The firm has", "The firm has not", "grown quickly."));
        assert_eq!(negate("It isn’t cheap").unwrap().flipped, "It is cheap.");
        let g = negate("Heavy objects do not fall faster").unwrap();
        assert_eq!((g.generic_stated.as_str(), g.generic_flipped.as_str()), ("They do not", "They do"));
        let g = negate("The firm will grow").unwrap();
        assert_eq!((g.generic_stated.as_str(), g.generic_flipped.as_str()), ("It will", "It will not"));
        assert_eq!(n("I have a dog"), None); // "have" as a main verb
        assert_eq!(n("They had to leave"), None);
        assert_eq!(n("Prices have fallen"), t("Prices have", "Prices have not", "fallen."));
        assert_eq!(n("Light bends in water"), None); // no auxiliary: no reliable flip
        assert_eq!(n("Is light fast?"), None);
        assert_eq!(n("Yes it is"), None);
    }

    fn negation(contrast: f32, confidence: Confidence) -> NegationTest {
        NegationTest { flipped: "x is not y".into(), stated_logp: 0.0, flipped_logp: -contrast, contrast, raw_contrast: contrast, baseline: 0.0, confidence }
    }

    #[test]
    fn the_negation_test_moves_and_caps_the_verdict() {
        let (_t, s) = settings();
        let panel = [member("believer", 0.2, Confidence::Medium), member("skeptic", 0.1, Confidence::High)];
        // the panel alone is undecided (margin 0.1); a clear negation test decides it
        assert_eq!(judge(&panel, None, &familiar(), &s, String::new()).stance, Stance::Mixed);
        let yes = judge(&panel, Some(&negation(0.9, Confidence::High)), &familiar(), &s, String::new());
        assert_eq!(yes.stance, Stance::Yes);
        assert!(yes.relied_on.contains(&"negation test".to_string()) && yes.reasoning.contains("x is not y"));
        assert_eq!(yes.confidence, Confidence::Medium); // capped by the believer it also relies on
        let no = judge(&panel, Some(&negation(-0.9, Confidence::High)), &familiar(), &s, String::new());
        assert_eq!(no.stance, Stance::No);
        let mut off = s.clone();
        off.council.negation_weight = 0.0;
        assert_eq!(judge(&panel, Some(&negation(-0.9, Confidence::High)), &familiar(), &off, String::new()).stance, Stance::Mixed);
    }

    #[test]
    fn a_trained_model_prefers_what_it_read_to_its_negation() {
        let (_tmp, s) = trained(150);
        let brain = Brain::load(&s).unwrap();
        let none = brain.prefix("", 40, None);
        let read = negation_test(&brain, &negate("Light is refracted when it passes from air into glass").unwrap(), &none, &s, None);
        assert!(read.contrast > 0.0, "{read:?}");
        let flipped = negation_test(&brain, &negate("Light is not refracted when it passes from air into glass").unwrap(), &none, &s, None);
        assert!((flipped.contrast + read.contrast).abs() < 1e-3, "{flipped:?} vs {read:?}");
    }

    #[test]
    fn judge_can_say_no_and_undecided() {
        let (_t, s) = settings();
        let no = judge(&[member("believer", 0.1, Confidence::High), member("skeptic", 0.9, Confidence::High)], None, &familiar(), &s, String::new());
        assert_eq!((no.stance, no.confidence, no.relied_on.clone()), (Stance::No, Confidence::High, vec!["skeptic".to_string()]));
        assert!(no.verdict.starts_with("No"));
        let mixed = judge(&[member("believer", 0.30, Confidence::High), member("skeptic", 0.25, Confidence::High)], None, &familiar(), &s, String::new());
        assert_eq!((mixed.stance, mixed.confidence), (Stance::Mixed, Confidence::Low));
        assert!(mixed.unresolved.is_some());
    }

    #[test]
    fn judge_is_capped_by_weakest_input_and_familiarity() {
        let (_t, s) = settings();
        let capped = judge(&[member("believer", 0.9, Confidence::Medium), member("skeptic", 0.0, Confidence::High)], None, &familiar(), &s, String::new());
        assert_eq!((capped.stance, capped.confidence, capped.confidence_before_cap), (Stance::Yes, Confidence::Medium, Some(Confidence::High)));
        let unfamiliar = Familiarity { cap: Confidence::Low, notes: vec!["the claim is unlike most of what it read".into()], ..familiar() };
        let low = judge(&[member("believer", 0.9, Confidence::High), member("skeptic", 0.0, Confidence::High)], None, &unfamiliar, &s, String::new());
        assert!(low.confidence == Confidence::Low && low.unresolved.unwrap().contains("unlike"));
    }

    #[test]
    fn investor_moves_the_margin() {
        let (_t, s) = settings();
        let panel = |inv: f32| vec![member("believer", 0.5, Confidence::High), member("skeptic", 0.3, Confidence::High), member("investor", inv, Confidence::High)];
        assert_eq!(judge(&panel(-1.0), None, &familiar(), &s, String::new()).stance, Stance::No);
        let yes = judge(&panel(0.8), None, &familiar(), &s, String::new());
        assert_eq!((yes.stance, yes.relied_on), (Stance::Yes, vec!["believer".to_string(), "investor".to_string()]));
    }

    #[test]
    fn consistency_check() {
        let (_t, s) = settings();
        let a = judge(&[member("believer", 0.9, Confidence::High), member("skeptic", 0.0, Confidence::High)], None, &familiar(), &s, String::new());
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
