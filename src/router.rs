//! Decides whether the Investor persona runs: keyword matching against
//! router.money_keywords in settings.yaml. Transparent and easy to extend.

use crate::config::Settings;
use crate::text::{keywords, words};
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct Routing {
    pub is_money_idea: bool,
    pub reason: String,
    /// The claim's distinctive words; the research loop counts them across
    /// sessions to find subjects worth reading up on.
    pub keywords: Vec<String>,
}

pub fn route(claim: &str, settings: &Settings) -> Routing {
    let lexicon: Vec<String> = settings.router.money_keywords.iter().map(|k| k.to_lowercase()).collect();
    let claim_words: std::collections::BTreeSet<String> = words(claim).into_iter().collect();
    let mut hits: Vec<String> = claim_words.into_iter().filter(|w| lexicon.contains(w)).collect();
    if lexicon.iter().any(|k| k == "$") && claim.contains('$') {
        hits.push("$".into());
    }
    let reason = if hits.is_empty() {
        "No money-related words, so no Investor.".to_string()
    } else {
        format!("Mentions {}.", hits.join(", "))
    };
    Routing { is_money_idea: !hits.is_empty(), reason, keywords: keywords(claim, 4) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::testing;

    #[test]
    fn routes_money_ideas_to_the_investor() {
        let tmp = tempfile::tempdir().unwrap();
        let s = testing::settings(tmp.path());
        let money = route("We should charge $9/month for a pro tier", &s);
        assert!(money.is_money_idea && money.reason == "Mentions charge, $.");
        let plain = route("The Great Wall of China is visible from orbit", &s);
        assert!(!plain.is_money_idea && plain.keywords.contains(&"visible".to_string()));
    }
}
