//! Basic speech: short, straight answers to the things people say to an
//! assistant (hello, thanks, who are you, how sure are you), by fixed rules
//! that only ever state facts about Aegist itself - never a made-up answer
//! to a question it can't check. And a voice, through the speech engine
//! the operating system already has (no AI model is involved in that).

use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

/// What Aegist knows about itself right now, for its answers.
#[derive(Clone, Debug, Default)]
pub struct Facts {
    /// The model, described (None: not trained yet).
    pub model: Option<String>,
    /// Apps it has learned to use.
    pub apps: Vec<String>,
}

fn normalize(text: &str) -> String {
    let t: String = text.to_lowercase().chars().map(|c| if c.is_alphanumeric() || c == '\'' || c == ' ' { c } else { ' ' }).collect();
    t.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// A reply to small talk, or None if it isn't small talk.
pub fn reply(text: &str, facts: &Facts) -> Option<String> {
    let t = normalize(text);
    let is = |opts: &[&str]| opts.contains(&t.as_str());
    let starts = |opts: &[&str]| opts.iter().any(|o| t == *o || t.starts_with(&format!("{o} ")));
    let has = |opts: &[&str]| opts.iter().any(|o| format!(" {t} ").contains(&format!(" {o} ")));
    if t.is_empty() {
        return None;
    }
    if is(&["hi", "hello", "hey", "yo", "hiya", "howdy", "good morning", "good afternoon", "good evening", "hi there", "hello there",
            "hey there", "sup", "what's up", "whats up"]) {
        return Some("Hi. What are we building?".into());
    }
    if starts(&["how are you", "how's it going", "hows it going", "how are things", "you ok", "are you ok"]) {
        return Some("Working, and ready. What do you need?".into());
    }
    if starts(&["thanks", "thank you", "thx", "ty", "cheers", "much appreciated"]) {
        return Some("You're welcome.".into());
    }
    if is(&["bye", "goodbye", "see you", "see ya", "later", "good night", "goodnight", "cya"]) {
        return Some("Bye. Type /exit to leave.".into());
    }
    if has(&["chatgpt", "gpt", "claude", "gemini", "llama", "jev", "copilot", "openai", "anthropic"]) && (t.starts_with("are you") || has(&["use", "using", "based"])) {
        return Some("No. I'm Aegist: my model is written and trained from scratch in this program. No other AI runs here - I \
                     borrow ideas (like typed decisions), never another model."
            .into());
    }
    if starts(&["who are you", "what are you", "what's your name", "whats your name", "what is your name", "introduce yourself"]) {
        let model = facts.model.as_deref().map(|m| format!(" Right now I'm a {m}.")).unwrap_or_else(|| " I haven't been trained yet.".into());
        return Some(format!(
            "I'm Aegist, a coding AI grown from scratch - my own transformer, trained on your machine, no other AI involved.{model} \
             I write, complete and fix code and check it before you get it, run programs, and learn to use apps on your screen."
        ));
    }
    if starts(&["what can you do", "what do you do", "help me", "can you help", "what are you good at", "what are your abilities"]) {
        let apps = if facts.apps.is_empty() { String::new() } else { format!(" Apps I've learned: {}.", facts.apps.join(", ")) };
        return Some(format!(
            "Code: write, complete and fix it, checked by the language's own tools and your tests. Run programs, tests, servers \
             and games; preview web pages. Decisions: pick from options with a real probability, or say I don't know. Apps: \
             learn a drawing app by trying its buttons, then draw what you ask and check the screen that it worked.{apps} \
             /help lists everything."
        ));
    }
    if starts(&["are you sure", "how sure are you", "can you be wrong", "do you hallucinate", "do you make mistakes", "do you lie",
                "can i trust you", "are you right", "is that right", "is that correct"]) {
        return Some("I can be wrong, so I check instead of asking you to trust me. Code goes through the language's own checker, \
                     a search for names I might have invented, and your tests; decisions come with their probability; actions in \
                     apps are checked on the screen. When the checks fail I say no, and when I have nothing to go on I say I \
                     don't know."
            .into());
    }
    if starts(&["are you smart", "are you intelligent", "are you conscious", "are you alive", "do you have feelings", "are you human"]) {
        return Some("I'm a small program with a model trained from scratch - useful at what I've learned, and honest about the \
                     rest. I'm not conscious and I don't have feelings."
            .into());
    }
    if starts(&["tell me a joke", "say something funny", "joke"]) {
        return Some("I only know one: it works on my machine. (I check that it works on yours.)".into());
    }
    None
}

/// Feedback in plain words: Some(true) for "good", Some(false) for "bad".
pub fn feedback(text: &str) -> Option<bool> {
    match normalize(text).as_str() {
        "good" | "right" | "correct" | "yes that's right" | "that's right" | "thats right" | "well done" | "good job" | "nice" | "perfect"
        | "that worked" | "it worked" | "great" | "/good" => Some(true),
        "bad" | "wrong" | "incorrect" | "no that's wrong" | "that's wrong" | "thats wrong" | "that didn't work" | "that did not work"
        | "it didn't work" | "nope" | "/bad" => Some(false),
        _ => None,
    }
}

static SPEAKER: Mutex<Option<Child>> = Mutex::new(None);

/// Text without colors or box drawing, for speaking.
pub fn plain(text: &str) -> String {
    let mut out = String::new();
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // skip an escape sequence
            for d in chars.by_ref() {
                if d.is_ascii_alphabetic() {
                    break;
                }
            }
        } else if c.is_alphanumeric() || c.is_whitespace() || ".,;:!?'\"()-%".contains(c) {
            out.push(c);
        }
    }
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The system's speech engine: which program, with which arguments (the
/// text goes on standard input).
fn engine() -> Option<(String, Vec<String>)> {
    if cfg!(windows) {
        let script = "Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; \
                      $s.Speak([Console]::In.ReadToEnd())";
        return Some(("powershell".into(), vec!["-NoProfile".into(), "-Command".into(), script.into()]));
    }
    if cfg!(target_os = "macos") {
        return Some(("say".into(), vec!["-f".into(), "-".into()]));
    }
    for (p, args) in [("espeak-ng", vec!["--stdin"]), ("espeak", vec!["--stdin"]), ("spd-say", vec!["-e"])] {
        if crate::proc::which(p).is_some() {
            return Some((p.into(), args.into_iter().map(String::from).collect()));
        }
    }
    None
}

/// Whether this system can speak.
pub fn can_speak() -> Result<(), String> {
    engine().map(|_| ()).ok_or_else(|| "there's no speech engine here: install espeak-ng (or speech-dispatcher) to hear me".into())
}

/// Say `text` aloud, cutting off whatever is being said now. Doesn't wait.
pub fn speak(text: &str) {
    let text = plain(text);
    if text.is_empty() {
        return;
    }
    let Some((prog, args)) = engine() else { return };
    let mut guard = SPEAKER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(mut old) = guard.take() {
        let _ = old.kill();
        let _ = old.wait();
    }
    let child = Command::new(prog).args(args).stdin(Stdio::piped()).stdout(Stdio::null()).stderr(Stdio::null()).spawn();
    if let Ok(mut child) = child {
        if let Some(mut stdin) = child.stdin.take() {
            use std::io::Write;
            let _ = stdin.write_all(text.as_bytes());
        }
        *guard = Some(child);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_talk_gets_short_true_answers_and_the_rest_is_left_alone() {
        let f = Facts { model: Some("4x192 transformer".into()), apps: vec!["paint".into()] };
        assert!(reply("Hello!", &f).unwrap().starts_with("Hi"));
        assert!(reply("thanks a lot", &f).unwrap().contains("welcome"));
        assert!(reply("who are you?", &f).unwrap().contains("4x192"));
        assert!(reply("are you chatgpt", &f).unwrap().starts_with("No."));
        assert!(reply("are you using jev?", &f).unwrap().starts_with("No."));
        assert!(reply("what can you do", &f).unwrap().contains("paint"));
        assert!(reply("do you hallucinate", &f).unwrap().contains("don't know"));
        assert_eq!(reply("write a snake game", &f), None);
        assert_eq!(reply("what is the capital of France?", &f), None);
        assert_eq!(feedback("Good job!"), Some(true));
        assert_eq!(feedback("that's wrong"), Some(false));
        assert_eq!(feedback("good code please"), None);
        assert_eq!(plain("\x1b[31m✗ No.\x1b[0m  it failed"), "No. it failed");
    }
}
