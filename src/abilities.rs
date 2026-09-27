//! The session's non-code abilities: basic speech, typed decisions, using
//! apps on the screen, and learning from your feedback on all of them.

use crate::agent::desktop::{self, Desktop};
use crate::agent::memory::{AppMemory, Effect};
use crate::agent::risk::Hands;
use crate::agent::sim::SimPaint;
use crate::agent::{self, Agent, Goal, Note};
use crate::decide::{self, Decider, Question};
use crate::predict::{self, Estimates};
use crate::session::{BrainState, Ctx};
use crate::talk;
use crate::ui::style::{self, pal, Style};
use crate::ui::{text, widgets};
use anyhow::{bail, Result};

/// What your "good" / "bad" applies to.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum LastAct {
    Agent { app: String },
    Decision,
}

/// Remember what "good" / "bad" would apply to (also across runs of the program).
fn set_last(ctx: &Ctx, act: LastAct) {
    let path = ctx.settings.data_path("memory").join("last-act.json");
    let _ = std::fs::write(path, serde_json::to_string(&act).unwrap_or_default());
    ctx.shared.lock().expect("lock").last_act = Some(act);
}

fn last(ctx: &Ctx) -> Option<LastAct> {
    let here = ctx.shared.lock().expect("lock").last_act.clone();
    here.or_else(|| serde_json::from_str(&std::fs::read_to_string(ctx.settings.data_path("memory").join("last-act.json")).ok()?).ok())
}

/// "on this" -> "On this"
fn cap(s: &str) -> String {
    let mut c = s.chars();
    c.next().map(|f| f.to_uppercase().chain(c).collect()).unwrap_or_default()
}

/// Print a reply (and speak it, when the voice is on).
pub fn say(ctx: &Ctx, color: style::Rgb, head: &str, body: &str) {
    let w = ctx.width().saturating_sub(6);
    let mut lines = vec![String::new()];
    let mut wrapped = text::wrap(body, w.saturating_sub(text::width(head) + 1)).into_iter();
    let first = wrapped.next().unwrap_or_default();
    lines.push(format!("  {} {}", Style::new().fg(color).bold().paint(head), first));
    for l in wrapped {
        lines.push(format!("  {}{l}", " ".repeat(text::width(head) + 1)));
    }
    ctx.print(lines);
    if ctx.shared.lock().expect("lock").voice {
        talk::speak(&format!("{head} {body}"));
    }
}

pub fn facts(ctx: &Ctx) -> talk::Facts {
    let model = match &ctx.shared.lock().expect("lock").brain {
        BrainState::Ready(b) => Some(format!("{:.1}M-parameter model trained from scratch", b.num_params as f64 / 1e6)),
        _ => None,
    };
    talk::Facts { model, apps: AppMemory::all(&ctx.settings).into_iter().map(|m| m.app).collect() }
}

/// Small talk and feedback, if that's what `text` is. Returns whether it handled it.
pub fn converse(ctx: &Ctx, text: &str) -> Result<bool> {
    if let Some(good) = talk::feedback(text) {
        feedback(ctx, good)?;
        return Ok(true);
    }
    if let Some(r) = talk::reply(text, &facts(ctx)) {
        let head = if r.starts_with("No.") { "" } else { "◆" };
        let (color, body) = if head.is_empty() { (pal::RED, r.trim_start_matches("No.").trim().to_string()) } else { (pal::CYAN, r) };
        say(ctx, color, if head.is_empty() { "No." } else { head }, &body);
        return Ok(true);
    }
    let lower = text.trim().to_lowercase();
    let first = lower.split_whitespace().next().unwrap_or("");
    if (matches!(first, "predict" | "forecast" | "extrapolate") || lower.starts_with("what comes next") || lower.starts_with("what's next")
        || lower.starts_with("next number")) && predict::numbers(text).len() >= 2 {
        predict_series(ctx, text)?;
        return Ok(true);
    }
    if first == "estimate" || lower.starts_with("guess how") {
        estimate(ctx, text.trim().split_once(char::is_whitespace).map_or("", |x| x.1))?;
        return Ok(true);
    }
    for lead in ["the answer was ", "the answer is ", "answer: ", "it was ", "correct answer: "] {
        if let Some(rest) = lower.strip_prefix(lead) {
            if matches!(last(ctx), Some(LastAct::Decision)) {
                answer(ctx, &text.trim()[lead.len()..][..rest.len()])?;
                return Ok(true);
            }
        }
    }
    if let Some((app, goal)) = app_request(text) {
        let app = app.or_else(|| match last(ctx) {
            Some(LastAct::Agent { app }) => Some(app),
            _ => None,
        });
        let Some(app) = app else {
            say(ctx, pal::YELLOW, "Which app?", "Say it with the app, like \"draw a red circle in paint\", or \"learn to use paint\". \
                                                  To practice without touching your desktop: \"learn to use the practice app\".");
            return Ok(true);
        };
        run_goal(ctx, &app, &goal, false)?;
        return Ok(true);
    }
    Ok(false)
}

fn is_sim(app: &str) -> bool {
    let a = app.trim().to_lowercase();
    matches!(a.as_str(), "sim" | "practice" | "practice app" | "the practice app" | "practice paint" | "--sim" | "the practice paint app")
}

/// "draw a red circle in paint" -> (Some("paint"), "draw a red circle");
/// "learn to use paint" -> (Some("paint"), "learn").
pub fn app_request(text: &str) -> Option<(Option<String>, String)> {
    let t = text.trim().trim_end_matches(['.', '!']);
    let lower = t.to_lowercase();
    for lead in ["learn to use ", "learn how to use ", "figure out ", "explore ", "study "] {
        if let Some(app) = lower.strip_prefix(lead) {
            let app = app.trim_start_matches("the ").trim_end_matches(" app").trim();
            return (!app.is_empty()).then(|| (Some(app.to_string()), "learn".to_string()));
        }
    }
    let first = lower.split_whitespace().next().unwrap_or("");
    if !matches!(first, "draw" | "sketch" | "fill" | "erase" | "practice" | "practise" | "clear" | "doodle" | "scribble") {
        return None;
    }
    // "clear" alone is the screen command's word; only "clear the canvas" is a goal
    if first == "clear" && !lower.contains("canvas") {
        return None;
    }
    match lower.rfind(" in ") {
        Some(i) if !lower[i + 4..].trim().is_empty() && agent::vision::parse_color(lower[i + 4..].trim()).is_none() => {
            let app = lower[i + 4..].trim().trim_start_matches("the ").trim_end_matches(" app").to_string();
            Some((Some(app), t[..i].to_string()))
        }
        _ => Some((None, t.to_string())),
    }
}

/// `/agent learn|do|practice|memory ...`
pub fn agent_command(ctx: &Ctx, args: &str) -> Result<()> {
    agent_command_for(ctx, args, None)
}

/// `agent_command`, with the app named separately (it may have spaces).
pub fn agent_command_for(ctx: &Ctx, args: &str, app_name: Option<&str>) -> Result<()> {
    let mut words: Vec<String> = args.split_whitespace().map(String::from).collect();
    let mut flag = |name: &str| -> bool {
        let had = words.iter().any(|w| w == name);
        words.retain(|w| w != name);
        had
    };
    let (sim, yes) = (flag("--sim"), flag("--yes"));
    let mut app = None;
    if let Some(i) = words.iter().position(|w| w == "--app") {
        if i + 1 < words.len() {
            app = Some(words.remove(i + 1));
        }
        words.remove(i);
    }
    let app = if sim { Some("practice".to_string()) } else { app_name.map(String::from).or(app) };
    let (sub, rest) = match words.split_first() {
        Some((s, r)) => (s.to_lowercase(), r.join(" ")),
        None => (String::new(), String::new()),
    };
    let last_app = || match last(ctx) {
        Some(LastAct::Agent { app }) => Some(app),
        _ => None,
    };
    match sub.as_str() {
        "" | "memory" | "apps" | "memories" => {
            memory(ctx, app.as_deref().or(if rest.is_empty() { None } else { Some(rest.as_str()) }));
            Ok(())
        }
        "learn" | "practice" | "do" => {
            let app = app.or_else(|| (sub == "learn" && !rest.is_empty()).then(|| rest.clone())).or_else(last_app);
            let Some(app) = app else { bail!("which app? /agent {sub} --app <name of its window>, or --sim for the practice app") };
            let goal = match sub.as_str() {
                "learn" => "learn".to_string(),
                "practice" => format!("practice {}", rest.trim()),
                _ => rest.clone(),
            };
            if goal.trim().is_empty() {
                bail!("/agent do <goal>, e.g. /agent do draw a red rectangle --app paint");
            }
            run_goal(ctx, &app, &goal, yes)
        }
        _ => {
            // "/agent draw a line in paint"
            let (a, goal) = app_request(args).unwrap_or((None, args.to_string()));
            let Some(app) = app.or(a).or_else(last_app) else { bail!("which app? add --app <name> (or --sim)") };
            run_goal(ctx, &app, &goal, yes)
        }
    }
}

fn note_line(n: &Note) -> String {
    match n {
        Note::Step(s) => format!("    {} {}", style::faint("·"), style::dim(s)),
        Note::Learned(s) => format!("    {} {}", style::fg(pal::GREEN, "+"), s),
        Note::Warn(s) => format!("    {} {}", style::fg(pal::YELLOW, "!"), s),
    }
}

/// Take `goal` to the app: understand it (or refuse it), then act, check and learn.
pub fn run_goal(ctx: &Ctx, app: &str, goal_text: &str, yes: bool) -> Result<()> {
    let goal = match agent::parse_goal(goal_text) {
        Ok(g) => g,
        Err(why) => {
            let (head, body) = match why.strip_prefix("No. ") {
                Some(b) => ("No.", b.to_string()),
                None => ("I don't know.", why.clone()),
            };
            say(ctx, if head == "No." { pal::RED } else { pal::YELLOW }, head, &body);
            return Ok(());
        }
    };
    let sim = is_sim(app);
    let mut real: Option<Box<dyn Desktop>> = None;
    let mut practice: Option<SimPaint> = None;
    let desk: &mut dyn Desktop = if sim {
        practice = Some(ctx.shared.lock().expect("lock").sim.take().unwrap_or_else(|| SimPaint::new(crate::util::unix_now() as u64 % 1000 + 1)));
        practice.as_mut().expect("just set")
    } else {
        real = Some(desktop::open()?);
        real.as_deref_mut().expect("just set")
    };
    let result = (|| -> Result<()> {
        let window = agent::find_window(desk, if sim { None } else { Some(app) })?;
        let risky_ok = yes || goal_text.to_lowercase().starts_with("yes");
        if !sim && ctx.settings.agent.confirm && !yes {
            let q = format!("Use {:?} on your screen? I'll only touch its window. Move the mouse to stop me.", window.title);
            if ctx.ask(&q, &[('y', "yes, go"), ('n', "no")]) != 'y' {
                say(ctx, pal::DIM, "Okay.", "I didn't touch anything.");
                return Ok(());
            }
        }
        ctx.print(vec![String::new(), format!("  {} {} {}", style::fg(pal::VIOLET, "◈"), Style::new().bold().paint(&goal.describe()),
                                               style::faint(&format!("· {}", window.title)))]);
        ctx.status(format!("{}…", goal.describe()));
        let mut note = |n: Note| ctx.line(note_line(&n));
        let mut hands = Hands::new(desk, window, &ctx.settings.agent);
        hands.cancel = Some(ctx.cancel.clone());
        let mut agent = Agent::new(hands, &ctx.settings, &mut note);
        let key = agent.memory.app.clone();
        let outcome = match &goal {
            Goal::Learn => agent.learn().map(|found| {
                let useful = found.iter().filter(|f| f.1 != Effect::Nothing).count();
                (true, format!("I tried {} buttons; {useful} do something I can use. Give me a goal, or say \"practice\".", found.len()))
            }),
            Goal::Practice(n) => agent.practice(*n).map(|(a, b)| {
                (true, format!("Practiced {n} goals I set myself: {:.0}% worked in the first half, {:.0}% in the second.", a * 100.0, b * 100.0))
            }),
            g => agent.attempt(g, risky_ok).map(|o| {
                if !o.tried {
                    (false, o.message)
                } else if o.worked {
                    (true, format!("Done: {}. (I expected it to work {:.0}% - say \"good\" or \"bad\" to teach me.)", o.message, o.expected * 100.0))
                } else {
                    (false, format!("That didn't work: {}. I've lowered my trust in the buttons I used.", o.message))
                }
            }),
        };
        let saved = agent.memory.save(&ctx.settings);
        let stopped_at = agent.last_seen.take();
        let shot = agent.hands.see().ok();
        drop(agent);
        set_last(ctx, LastAct::Agent { app: if sim { "practice".into() } else { app.to_string() } });
        saved?;
        let _ = key;
        match outcome {
            Ok((true, msg)) => say(ctx, pal::GREEN, "✓", &msg),
            Ok((false, msg)) if msg.starts_with("No") => say(ctx, pal::RED, "No.", msg.trim_start_matches("No:").trim_start_matches("No.").trim()),
            Ok((false, msg)) if msg.starts_with("I don't know") || msg.starts_with("I haven't") => say(ctx, pal::YELLOW, "I don't know.", &msg),
            Ok((false, msg)) => say(ctx, pal::RED, "✗", &msg),
            Err(e) => {
                let mut msg = e.to_string();
                if let Some(img) = stopped_at {
                    let path = ctx.settings.data_path("memory").join("stopped-here.png");
                    if img.save_png(&path).is_ok() {
                        msg.push_str(&format!(". What I saw when I stopped: {}", path.display()));
                    }
                }
                say(ctx, pal::YELLOW, "■", &msg)
            }
        }
        if sim {
            if let Some(img) = shot {
                let path = ctx.settings.data_path("memory").join("practice-screen.png");
                if img.save_png(&path).is_ok() {
                    if let Ok(lines) = crate::ide::render_png(&path, ctx.width().saturating_sub(4).min(72), 24) {
                        ctx.print(lines.into_iter().map(|l| format!("  {l}")).collect());
                    }
                }
            }
        }
        Ok(())
    })();
    if let Some(p) = practice {
        ctx.shared.lock().expect("lock").sim = Some(p);
    }
    drop(real);
    result
}

/// What it knows about each app.
pub fn memory(ctx: &Ctx, app: Option<&str>) {
    let all = AppMemory::all(&ctx.settings);
    let all: Vec<AppMemory> = match app {
        Some(a) => all.into_iter().filter(|m| m.app.contains(&crate::agent::memory::app_key(a)) || (is_sim(a) && m.app.contains("practice"))).collect(),
        None => all,
    };
    let mut lines = vec![String::new()];
    if all.is_empty() {
        lines.push(format!("  {} {}", style::fg(pal::YELLOW, "◇"), "I haven't learned any apps yet."));
        lines.push(format!("    {}", style::dim("Open one and say \"learn to use <app>\", or practice safely: \"learn to use the practice app\".")));
    }
    for m in all {
        let rate = m.success_rate(50).map(|r| format!(" · {:.0}% of the last goals worked", r * 100.0)).unwrap_or_default();
        let cal = m.calibration(50).map(|b| format!(" · confidence error {b:.2} (0 is perfect)")).unwrap_or_default();
        lines.push(format!("  {} {}  {}", style::gradient_styled(&m.app, 0.2, true), style::faint(&format!("{} attempts{rate}{cal}", m.attempts.len())), ""));
        let mut known: Vec<_> = m.elements.iter().filter(|k| k.effect != Effect::Nothing).collect();
        known.sort_by_key(|k| format!("{:?}", k.effect));
        for k in known {
            let trust = k.belief();
            lines.push(format!("    {} {}  {} {}", widgets::bar(trust, 8), text::pad(&k.effect.describe(), 28),
                style::faint(&format!("at ({}, {})", k.rect.x, k.rect.y)),
                style::faint(&format!("· {} worked, {} failed{}", k.successes, k.failures, if k.effect.risky() { " · risky" } else { "" }))));
        }
        if m.undo_works == Some(false) {
            lines.push(format!("    {} {}", style::fg(pal::YELLOW, "!"), style::dim("undo didn't always put the canvas back")));
        }
    }
    lines.push(String::new());
    ctx.print(lines);
}

/// You said whether the last thing was right.
pub fn feedback(ctx: &Ctx, good: bool) -> Result<()> {
    match last(ctx) {
        Some(LastAct::Agent { app }) => {
            let key = if is_sim(&app) { "practice".to_string() } else { app };
            let mut m = AppMemory::all(&ctx.settings).into_iter().find(|m| m.app.contains(&crate::agent::memory::app_key(&key)))
                .ok_or_else(|| anyhow::anyhow!("I have no memory of {key} to correct"))?;
            let Some(a) = m.feedback(good) else { bail!("I haven't tried anything in {} yet", m.app) };
            m.save(&ctx.settings)?;
            let saw = if a.worked { "it looked right to me" } else { "it looked wrong to me" };
            let msg = if a.worked == good {
                format!("Noted - \"{}\": {saw} too. That button's record is stronger now.", a.goal)
            } else {
                format!("Noted - \"{}\": {saw}, but you say otherwise, so I've corrected its record. You see the screen better than I do.", a.goal)
            };
            say(ctx, pal::GREEN, "✓", &msg);
        }
        Some(LastAct::Decision) => {
            let log = decide::Log::new(&ctx.settings);
            let d = log.feedback(None, good)?;
            let learned = log.learn()?;
            let how = if learned.judged >= 3 {
                format!(" From {} judged decisions ({:.0}% right) my confidence scale is now {:.2} (was {:.2}); calibration error {:.2} → {:.2}.",
                    learned.judged, learned.accuracy * 100.0, learned.after.temperature, learned.before.temperature, learned.error_before, learned.error_after)
            } else {
                format!(" After {} more judged decisions I'll start adjusting how sure I let myself be.", 3 - learned.judged)
            };
            say(ctx, pal::GREEN, "✓", &format!("Noted: decision #{} was {}.{how}", d.id, if good { "right" } else { "wrong" }));
        }
        None => say(ctx, pal::YELLOW, "I don't know", "what that's about: I haven't done anything this session to give feedback on."),
    }
    Ok(())
}

/// `/decide <question>: a | b | c` (no options: is it true?)
pub fn decide(ctx: &Ctx, args: &str) -> Result<()> {
    let args = args.trim();
    if args.is_empty() {
        bail!("/decide <question>: <option> | <option> | ...   or   /decide <statement> (is it true?)");
    }
    let q = if args.contains('|') {
        let (question, opts) = match args.rsplit_once(':') {
            Some((q, o)) if o.contains('|') => (q.trim(), o),
            _ => ("which one?", args),
        };
        let options: Vec<&str> = opts.split('|').map(str::trim).filter(|o| !o.is_empty()).collect();
        if options.len() < 2 {
            bail!("give at least two options, separated by |");
        }
        Question::choose(question, &options)
    } else {
        Question::truth(args.trim_end_matches('?'))
    };
    decide_question(ctx, q)
}

/// Decide `q` with the model and show the result.
pub fn decide_question(ctx: &Ctx, q: Question) -> Result<()> {
    // no trained model: decide from experience alone
    let brain = ctx.brain().ok();
    ctx.status("Deciding…");
    let decider = Decider::new(brain.as_deref(), &ctx.settings);
    let d = decider.decide(q);
    set_last(ctx, LastAct::Decision);
    let mut lines = vec![String::new(), format!("  {} {}  {}", style::fg(pal::VIOLET, "◈"), Style::new().bold().paint(&d.question.text),
                                                    style::faint(&format!("#{} · {:.0} ms", d.id, d.millis)))];
    let pad = d.question.options.iter().map(|o| text::width(o)).max().unwrap_or(4).min(30);
    for (i, (o, p)) in d.question.options.iter().zip(&d.probs).enumerate() {
        let mark = if d.pick == Some(i) { style::fg(pal::GREEN, "❯") } else { " ".into() };
        lines.push(format!("    {mark} {}  {} {}", text::pad(&text::truncate(o, 30), pad), widgets::bar(*p, 16), style::dim(&format!("{:>3.0}%", p * 100.0))));
    }
    let lessons = decider.experience.lessons;
    let parts = match (decider.brain.is_some(), lessons > 0) {
        (true, true) => format!("from the model and from {lessons} lessons of your feedback"),
        (true, false) => "from the model (no feedback to learn from yet)".into(),
        (false, true) => format!("from {lessons} lessons of your feedback (no trained model)"),
        (false, false) => "no trained model and no feedback yet, so nothing to go on".into(),
    };
    lines.push(format!("    {}", style::faint(&parts)));
    ctx.print(lines);
    match d.pick {
        Some(_) => say(ctx, pal::GREEN, d.answer(), &format!("- {}. Say \"good\" or \"bad\" to teach me.", d.reason)),
        None => say(ctx, pal::YELLOW, "I don't know:", &format!("{}. I won't pick one just to have an answer.", d.reason)),
    }
    Ok(())
}

/// The right answer to the last decision.
pub fn answer(ctx: &Ctx, option: &str) -> Result<()> {
    let log = decide::Log::new(&ctx.settings);
    let d = log.answer(option)?;
    let was_right = d.pick.is_some_and(|i| d.question.options[i].eq_ignore_ascii_case(option.trim()));
    let msg = match (d.pick, was_right) {
        (Some(_), true) => "that's what I picked. Noted, I'll lean that way more firmly next time.".to_string(),
        (Some(i), false) => format!("I picked {:?}. I've learned from it: similar questions will lean to {:?}.", d.question.options[i], option.trim()),
        (None, _) => format!("I didn't know. Now I've learned it: similar questions will lean to {:?}.", option.trim()),
    };
    say(ctx, pal::GREEN, "✓", &format!("Decision #{}: {msg}", d.id));
    Ok(())
}

/// `/predict 3 5 7 9 [next 3]`: the next values of a series.
pub fn predict_series(ctx: &Ctx, args: &str) -> Result<()> {
    let lower = args.to_lowercase();
    let mut xs = predict::numbers(args);
    let mut steps = 1;
    if let Some(i) = lower.find("next ").or_else(|| lower.find("--steps ")) {
        if let Some(n) = predict::numbers(&lower[i..]).first() {
            steps = (*n as usize).clamp(1, 50);
            // that number isn't part of the series
            if let Some(pos) = xs.iter().rposition(|x| x == n) {
                xs.remove(pos);
            }
        }
    }
    let mut lines = vec![String::new(), format!("  {} {}  {}", style::fg(pal::VIOLET, "◈"),
        Style::new().bold().paint(&xs.iter().map(|x| predict::fmt(*x)).collect::<Vec<_>>().join(", ")),
        widgets::sparkline(&xs.iter().map(|x| *x as f32).collect::<Vec<_>>(), 24))];
    match predict::forecast(&xs, steps) {
        Err(why) => {
            ctx.print(lines);
            say(ctx, pal::YELLOW, "I don't know.", &cap(why.trim_start_matches("I don't know: ")));
        }
        Ok(f) => {
            for (k, (v, lo, hi)) in f.steps.iter().enumerate() {
                lines.push(format!("    {} {}  {}", style::faint(&format!("+{}", k + 1)), Style::new().fg(pal::CYAN).bold().paint(&predict::fmt(*v)),
                    style::dim(&format!("likely between {} and {}", predict::fmt(*lo), predict::fmt(*hi)))));
            }
            lines.push(format!("    {}", style::faint(&format!("because {} · tested on {} past points, off by {} on average",
                f.method.describe(), f.tested, predict::fmt(f.past_error)))));
            ctx.print(lines);
            let (v, lo, hi) = f.steps[0];
            say(ctx, pal::GREEN, "◆", &format!("Next: {} (80% range {} to {}).", predict::fmt(v), predict::fmt(lo), predict::fmt(hi)));
        }
    }
    Ok(())
}

/// `/estimate <thing>` from similar things you've told it; `/estimate <thing> = <number>` teaches it.
pub fn estimate(ctx: &Ctx, args: &str) -> Result<()> {
    let est = Estimates::new(&ctx.settings);
    if let Some((what, value)) = args.rsplit_once('=') {
        let Some(v) = predict::numbers(value).first().copied() else { bail!("{:?} isn't a number", value.trim()) };
        est.teach(what, v)?;
        let n = est.all().iter().filter(|k| predict::similarity(&k.what, what) >= 0.34).count();
        say(ctx, pal::GREEN, "✓", &format!("Learned: {} = {}. I know {n} value{} like it now.", what.trim(), predict::fmt(v), if n == 1 { "" } else { "s" }));
        return Ok(());
    }
    if args.trim().is_empty() {
        bail!("/estimate <thing>   or teach me:   /estimate <thing> = <number>");
    }
    match est.estimate(args) {
        Err(why) => say(ctx, pal::YELLOW, "I don't know.", &cap(why.trim_start_matches("I don't know: "))),
        Ok(e) => {
            let mut lines = vec![String::new(), format!("  {} {}", style::fg(pal::VIOLET, "◈"), Style::new().bold().paint(args.trim()))];
            for (k, s) in e.from.iter().take(5) {
                lines.push(format!("    {} {} = {}", style::faint(&format!("{:>3.0}% alike", s * 100.0)), k.what, predict::fmt(k.value)));
            }
            ctx.print(lines);
            say(ctx, pal::GREEN, "◆", &format!("About {} (80% range {} to {}), from {} similar thing{} you told me.", predict::fmt(e.middle),
                predict::fmt(e.low), predict::fmt(e.high), e.from.len(), if e.from.len() == 1 { "" } else { "s" }));
        }
    }
    Ok(())
}

/// `/voice on|off`
pub fn voice(ctx: &Ctx, args: &str) -> Result<()> {
    let on = match args.trim() {
        "" => !ctx.shared.lock().expect("lock").voice,
        "on" | "yes" | "true" => true,
        "off" | "no" | "false" => false,
        other => bail!("/voice on or /voice off (not {other:?})"),
    };
    if on {
        if let Err(why) = talk::can_speak() {
            bail!("I can't speak here: {why}");
        }
    }
    ctx.shared.lock().expect("lock").voice = on;
    say(ctx, pal::CYAN, "◆", if on { "Voice on. I'll say my answers out loud." } else { "Voice off." });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn app_requests_are_recognized() {
        assert_eq!(app_request("draw a red circle in paint"), Some((Some("paint".into()), "draw a red circle".into())));
        assert_eq!(app_request("learn to use the practice app"), Some((Some("practice".into()), "learn".into())));
        assert_eq!(app_request("draw a line"), Some((None, "draw a line".into())));
        assert_eq!(app_request("fill it in light blue"), Some((None, "fill it in light blue".into())));
        assert_eq!(app_request("clear"), None);
        assert_eq!(app_request("write a snake game"), None);
        assert!(is_sim("practice") && !is_sim("paint"));
    }
}
