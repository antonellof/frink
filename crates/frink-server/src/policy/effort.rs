//! Reasoning-effort dialect handling.
//!
//! Each checkpoint's chat template accepts only its own effort
//! vocabulary and hard-fails on the rest, while clients speak whatever
//! dialect their provider taught them. Named levels project onto the
//! numeric scale the OpenAI-compatible servers share, and
//! out-of-vocabulary values
//! quantize to the nearest supported gear instead of failing the
//! request.
//!
//! The vocabulary is *probed* from the checkpoint's own template, never
//! read from a static table keyed by model family: a family registry
//! cannot be keyed correctly, because two checkpoints that resolve to
//! the same parser can differ on whether they grade effort at all.
//! `probe_effort_profile` renders a handful of probe conversations
//! through the template and learns the answer from what moved and what
//! raised.
//!
//! Ported 1:1 from FreeToken's `python/freetoken/tokenizer/effort.py`
//! (Apache-2.0); see `docs/THIRD_PARTY_NOTICES.md`.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Map, Value};

/// One gear on the shared effort scale.
///
/// The numeric positions must match the table the other
/// OpenAI-compatible servers use, so the ecosystem agrees on what
/// "medium" means relative to "xhigh".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Effort {
    None,
    Minimal,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

/// Every gear, in scale order. `Max` and `XHigh` deliberately share a
/// position (0.99); see [`quantize_effort`] for why that is not a
/// problem.
pub const KNOWN_REASONING_EFFORTS: [Effort; 7] = [
    Effort::None,
    Effort::Minimal,
    Effort::Low,
    Effort::Medium,
    Effort::High,
    Effort::XHigh,
    Effort::Max,
];

/// OpenAI's effort triple -- the common-denominator vocabulary every
/// dialect understands.
pub const OPENAI_EFFORT_TRIPLE: [Effort; 3] = [Effort::Low, Effort::Medium, Effort::High];

/// The graded ladder served to a template that grades effort *without*
/// validating it. Such a template interpolates any string, so its
/// probed `supported` set is the whole scale -- but the checkpoint's
/// trained levels are the ladder names, and never-advertised dialects
/// (`minimal` / `none` / `max`) must quantize onto them instead of
/// reaching the model verbatim.
pub const GRADED_EFFORT_LADDER: [Effort; 4] =
    [Effort::Low, Effort::Medium, Effort::High, Effort::XHigh];

impl Effort {
    /// Position on the shared numeric scale.
    pub fn scale(self) -> f32 {
        match self {
            Effort::None => 0.0,
            Effort::Minimal => 0.1,
            Effort::Low => 0.2,
            Effort::Medium => 0.7,
            Effort::High => 0.9,
            Effort::XHigh => 0.99,
            Effort::Max => 0.99,
        }
    }

    /// The wire spelling, as a client sends it and as a template reads
    /// it.
    pub fn as_str(self) -> &'static str {
        match self {
            Effort::None => "none",
            Effort::Minimal => "minimal",
            Effort::Low => "low",
            Effort::Medium => "medium",
            Effort::High => "high",
            Effort::XHigh => "xhigh",
            Effort::Max => "max",
        }
    }

    /// Parse a client's spelling. An unknown string is not an error
    /// here -- it is a value with no position on the scale, which
    /// [`quantize_effort`] resolves to "send nothing".
    pub fn parse(value: &str) -> Option<Effort> {
        KNOWN_REASONING_EFFORTS
            .iter()
            .copied()
            .find(|e| e.as_str() == value)
    }
}

/// What one checkpoint's template accepts, learned by probing it.
///
/// `default` is the supported gear whose rendering is byte-identical to
/// passing no effort at all. `consumes_effort == false` means no probe
/// round ever changed the template's output or raised -- the template
/// ignores the knob, so requests should not carry it. `validates ==
/// true` means the probe observed a rejection: only then is `supported`
/// a real vocabulary rather than "this template interpolates anything".
/// `strength_dialect == true` means the template reads the
/// `reasoning_strength` spelling (a family whose documented ladder tops
/// out at `xhigh`) rather than only `reasoning_effort`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffortProfile {
    pub supported: BTreeSet<Effort>,
    pub default: Option<Effort>,
    pub consumes_effort: bool,
    pub validates: bool,
    pub strength_dialect: bool,
}

impl EffortProfile {
    /// The profile of a template that ignores effort entirely: nothing
    /// is supported and nothing should be sent.
    pub fn inert() -> Self {
        EffortProfile {
            supported: BTreeSet::new(),
            default: None,
            consumes_effort: false,
            validates: false,
            strength_dialect: false,
        }
    }
}

/// A checkpoint's thinking controls, learned by probing its template.
///
/// `toggleable`: the off/on broadcasts render differently.
/// `has_adaptive`: `thinking_mode: "adaptive"` is a third distinct
/// state. `default_state`: which state the bare render matches (`On`
/// when not toggleable, or when the bare render matches neither).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkingProfile {
    pub efforts: EffortProfile,
    pub toggleable: bool,
    pub has_adaptive: bool,
    pub default_state: ThinkingState,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingState {
    On,
    Off,
    Adaptive,
}

/// Protocol-level thinking toggles, broadcast in every spelling the
/// ecosystem's templates read (`enable_thinking` bool: qwen / glm /
/// gemma / dsv4; `thinking_mode` string: minimax-m3). A Jinja template
/// ignores undeclared variables, so each one simply picks the knob it
/// knows -- no per-family routing needed.
pub fn thinking_off_kwargs() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("enable_thinking".into(), json!(false));
    m.insert("thinking_mode".into(), json!("disabled"));
    m
}

pub fn thinking_on_kwargs() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("enable_thinking".into(), json!(true));
    m.insert("thinking_mode".into(), json!("enabled"));
    m
}

pub fn thinking_adaptive_kwargs() -> Map<String, Value> {
    let mut m = Map::new();
    m.insert("thinking_mode".into(), json!("adaptive"));
    m
}

/// The vocabulary a checkpoint is actually served (and advertised)
/// with.
///
/// A *narrowed* `supported` set is authoritative -- narrowing only ever
/// comes from observed rejections. A grader that accepted the whole
/// scale validated nothing, so pass-through would interpolate
/// off-vocabulary values (`minimal`, `max`) into the prompt verbatim;
/// it is capped to its dialect's ladder instead. The strength dialect's
/// documented levels top out at `xhigh`; a plain effort grader keeps the
/// OpenAI triple, the only vocabulary such a template is known to
/// understand.
pub fn effective_efforts(profile: &EffortProfile) -> BTreeSet<Effort> {
    if !profile.consumes_effort {
        return BTreeSet::new();
    }
    let all: BTreeSet<Effort> = KNOWN_REASONING_EFFORTS.iter().copied().collect();
    if profile.validates || profile.supported != all {
        return profile.supported.clone();
    }
    let ladder: &[Effort] = if profile.strength_dialect {
        &GRADED_EFFORT_LADDER
    } else {
        &OPENAI_EFFORT_TRIPLE
    };
    let mut vocab: BTreeSet<Effort> = ladder
        .iter()
        .copied()
        .filter(|e| profile.supported.contains(e))
        .collect();
    if let Some(default) = profile.default {
        vocab.insert(default);
    }
    vocab
}

/// A gear farther than this on the scale misrepresents the request;
/// drop the value and let the template default apply. Keeps OpenAI's
/// `medium` from escalating to an encoder's absolute-maximum `high` gear
/// (0.2 away), while `high` still reaches `xhigh` (0.09 away).
const MAX_QUANTIZE_DISTANCE: f32 = 0.15;

/// Map a client's effort onto `profile`; `None` means "send nothing".
///
/// In-vocabulary values pass through untouched. Other named levels land
/// on the nearest supported gear within [`MAX_QUANTIZE_DISTANCE`] --
/// except `max`, which is reachable only by its own name: an extreme
/// opt-in gear must never be entered by rounding. Everything else drops
/// to the template default. With `max` excluded the remaining scale
/// positions are unique, so quantization is deterministic across
/// processes.
pub fn quantize_effort(value: &str, profile: &EffortProfile) -> Option<Effort> {
    let supported = effective_efforts(profile);
    if supported.is_empty() {
        return None;
    }
    let asked = Effort::parse(value)?;
    if supported.contains(&asked) {
        return Some(asked);
    }
    let position = asked.scale();
    let mut ranked: Vec<Effort> = supported
        .iter()
        .copied()
        .filter(|e| *e != Effort::Max)
        .collect();
    // Nearest gear wins; a tie goes to the higher one, matching the
    // `(distance, -scale)` sort key upstream.
    ranked.sort_by(|a, b| {
        let da = (a.scale() - position).abs();
        let db = (b.scale() - position).abs();
        da.total_cmp(&db)
            .then_with(|| b.scale().total_cmp(&a.scale()))
    });
    match ranked.first() {
        Some(best) if (best.scale() - position).abs() <= MAX_QUANTIZE_DISTANCE => Some(*best),
        _ => None,
    }
}

/// One tool flips tool-conditional paths: an encoder may grade effort
/// only in thinking mode, which tools force.
fn probe_tools() -> Vec<Value> {
    vec![json!({
        "type": "function",
        "function": {
            "name": "noop",
            "description": "No-op probe tool.",
            "parameters": {"type": "object", "properties": {}},
        },
    })]
}

/// One probe: the `chat_template_kwargs` to render with, and the tools
/// to offer alongside them.
type ProbeRound<'a> = (Map<String, Value>, Option<&'a [Value]>);

/// Learn a checkpoint's effort vocabulary by rendering probes through
/// it.
///
/// `render(chat_template_kwargs, tools)` returns a comparable rendering
/// and errors on rejection. A round whose no-effort baseline errors is
/// skipped: the template rejected the probe *conversation shape*, not
/// the effort.
pub fn probe_effort_profile<E>(
    mut render: impl FnMut(&Map<String, Value>, Option<&[Value]>) -> Result<String, E>,
) -> EffortProfile {
    let tools = probe_tools();
    // Templates read effort unconditionally, under tool-forced
    // thinking, or only under an explicit thinking opt-in -- one round
    // per shape.
    let rounds: [ProbeRound; 3] = [
        (Map::new(), None),
        (Map::new(), Some(tools.as_slice())),
        (
            {
                let mut m = Map::new();
                m.insert("enable_thinking".into(), json!(true));
                m
            },
            None,
        ),
    ];

    let mut rejected: BTreeSet<Effort> = BTreeSet::new();
    let mut diverged: BTreeSet<Effort> = BTreeSet::new();
    let mut matches_baseline: Vec<(Effort, bool)> =
        KNOWN_REASONING_EFFORTS.iter().map(|e| (*e, true)).collect();
    let mut ran_rounds = 0usize;

    for (base_kwargs, round_tools) in rounds.iter() {
        let baseline = match render(base_kwargs, *round_tools) {
            Ok(text) => text,
            // The template rejects this probe shape, not the effort.
            Err(_) => continue,
        };
        ran_rounds += 1;
        for effort in KNOWN_REASONING_EFFORTS {
            let mut kwargs = base_kwargs.clone();
            kwargs.insert("reasoning_effort".into(), json!(effort.as_str()));
            match render(&kwargs, *round_tools) {
                // Any error means "not accepted".
                Err(_) => {
                    rejected.insert(effort);
                    set_match(&mut matches_baseline, effort, false);
                }
                Ok(rendering) => {
                    if rendering != baseline {
                        diverged.insert(effort);
                        set_match(&mut matches_baseline, effort, false);
                    }
                }
            }
        }
    }

    if ran_rounds == 0 {
        // Nothing learnable: sending no effort is the only safe
        // rendering.
        return EffortProfile::inert();
    }

    // Which spelling does the template read? A render that moves on the
    // `reasoning_strength` kwarg alone marks the graded-strength
    // dialect, whose ladder differs.
    let strength_dialect = match (render(&Map::new(), None), {
        let mut m = Map::new();
        m.insert("reasoning_strength".into(), json!("low"));
        render(&m, None)
    }) {
        (Ok(baseline), Ok(moved)) => moved != baseline,
        // A rejecting template is not this dialect.
        _ => false,
    };

    let supported: BTreeSet<Effort> = KNOWN_REASONING_EFFORTS
        .iter()
        .copied()
        .filter(|e| !rejected.contains(e))
        .collect();
    let consumes = !rejected.is_empty() || !diverged.is_empty();
    let mut default = None;
    if consumes {
        default = supported
            .iter()
            .copied()
            .filter(|e| matches(&matches_baseline, *e))
            // The bare render can match several gears; the highest of
            // them is the one the template actually defaults to.
            .max_by(|a, b| a.scale().total_cmp(&b.scale()));
    }
    EffortProfile {
        supported,
        default,
        consumes_effort: consumes,
        validates: !rejected.is_empty(),
        strength_dialect,
    }
}

/// Learn a checkpoint's thinking-toggle behaviour by rendering the
/// broadcast kwargs through its template. Same contract as
/// [`probe_effort_profile`]; a template that rejects any toggle probe is
/// treated as not toggleable.
pub fn probe_thinking_profile<E>(
    mut render: impl FnMut(&Map<String, Value>, Option<&[Value]>) -> Result<String, E>,
    efforts: EffortProfile,
) -> ThinkingProfile {
    let inert = |efforts: EffortProfile| ThinkingProfile {
        efforts,
        toggleable: false,
        has_adaptive: false,
        default_state: ThinkingState::On,
    };
    let (baseline, off, on) = match (
        render(&Map::new(), None),
        render(&thinking_off_kwargs(), None),
        render(&thinking_on_kwargs(), None),
    ) {
        (Ok(baseline), Ok(off), Ok(on)) => (baseline, off, on),
        // Can't observe the toggle; assume none.
        _ => return inert(efforts),
    };
    let toggleable = off != on;
    let mut has_adaptive = false;
    let mut adaptive: Option<String> = None;
    if toggleable {
        match render(&thinking_adaptive_kwargs(), None) {
            Ok(text) => {
                has_adaptive = text != off && text != on;
                adaptive = Some(text);
            }
            // Adaptive is not a state this template knows.
            Err(_) => has_adaptive = false,
        }
    }
    let mut default_state = ThinkingState::On;
    if toggleable {
        if baseline == off {
            default_state = ThinkingState::Off;
        } else if has_adaptive && adaptive.as_deref() == Some(baseline.as_str()) {
            default_state = ThinkingState::Adaptive;
        }
    }
    ThinkingProfile {
        efforts,
        toggleable,
        has_adaptive,
        default_state,
    }
}

/// The thinking mode a chat request resolves to.
///
/// One source of truth for the decision, because two sides depend on
/// it and must not disagree: the encode side picks the prompt the model
/// sees, and the parse side uses it to decide whether the model's
/// output *begins inside* a reasoning block (some encoders open the
/// block in the prompt and never emit its opening tag).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingMode {
    Chat,
    Thinking,
}

impl ThinkingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            ThinkingMode::Chat => "chat",
            ThinkingMode::Thinking => "thinking",
        }
    }
}

/// Resolve the thinking mode for a chat request.
///
/// Thinking is on when tools are offered -- some encoders emit
/// well-formed tool calls only in thinking mode -- or when the caller
/// asked for it through `chat_template_kwargs`. An unrecognized
/// `thinking_mode` string falls back to `Chat` rather than reaching the
/// template.
pub fn resolve_thinking_mode(
    chat_template_kwargs: Option<&Map<String, Value>>,
    tools: Option<&[Value]>,
) -> ThinkingMode {
    let empty = Map::new();
    let ctk = chat_template_kwargs.unwrap_or(&empty);
    let mut mode = match ctk.get("thinking_mode").and_then(|v| v.as_str()) {
        Some("thinking") => ThinkingMode::Thinking,
        _ => ThinkingMode::Chat,
    };
    let truthy = |key: &str| match ctk.get(key) {
        Some(Value::Bool(b)) => *b,
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Number(n)) => n.as_f64().map(|f| f != 0.0).unwrap_or(false),
        _ => false,
    };
    if tools.is_some_and(|t| !t.is_empty()) || truthy("enable_thinking") || truthy("thinking") {
        mode = ThinkingMode::Thinking;
    }
    mode
}

/// What `sanitize_effort` did to a request's `reasoning_effort`, so the
/// caller can log a mapping once instead of per request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffortMapping {
    /// The value was already in the checkpoint's vocabulary.
    Unchanged,
    /// Quantized onto the nearest supported gear.
    Mapped(Effort),
    /// Dropped: no gear is close enough, so the template default
    /// applies.
    Dropped,
}

/// Quantize a request's `reasoning_effort` onto what this checkpoint
/// accepts, in place.
///
/// Every render path -- the worker, request validation, token counting
/// -- must quantize identically, or a request validates against one
/// prompt and generates from another. Absent means absent: a request
/// that carried no effort is returned untouched.
pub fn sanitize_effort(
    chat_template_kwargs: &mut Map<String, Value>,
    profile: &EffortProfile,
) -> EffortMapping {
    let raw = match chat_template_kwargs.get("reasoning_effort") {
        Some(value) => value.clone(),
        None => return EffortMapping::Unchanged,
    };
    let asked = raw.as_str().unwrap_or("");
    let mapped = quantize_effort(asked, profile);
    match mapped {
        Some(effort) if effort.as_str() == asked => EffortMapping::Unchanged,
        Some(effort) => {
            chat_template_kwargs.insert("reasoning_effort".into(), json!(effort.as_str()));
            EffortMapping::Mapped(effort)
        }
        None => {
            chat_template_kwargs.remove("reasoning_effort");
            EffortMapping::Dropped
        }
    }
}

/// What a checkpoint can be *asked for*, as a client would list it.
///
/// One flat vocabulary rather than two fields, because a client picking
/// a control does not care whether "off" is a toggle and "high" is a
/// gear -- both are things it may send, and sending either is one
/// string in `reasoning_effort`.
///
/// `kwargs` is what makes the list usable without per-family knowledge:
/// it says, for each gear, exactly what `chat_template_kwargs` selects
/// it. A UI that has only the names still has to know that "off" means
/// broadcasting two booleans and "high" means a string, which is the
/// family knowledge this whole module exists to remove.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ThinkGears {
    /// In the order a UI would show them: least thinking first.
    pub supported: Vec<String>,
    /// Which one the checkpoint is already in when asked for nothing.
    /// Always a member of `supported`.
    pub default: Option<String>,
    /// Per gear, the `chat_template_kwargs` that select it. Keyed by the
    /// same strings as `supported`.
    pub kwargs: BTreeMap<String, Map<String, Value>>,
}

impl ThinkGears {
    /// A checkpoint that says nothing about thinking. Advertising an
    /// empty list would claim it was asked and answered "none"; this is
    /// the difference between "no gears" and "not a reasoning model".
    pub fn is_empty(&self) -> bool {
        self.supported.is_empty()
    }

    /// The kwargs that select `gear`, if it is on offer.
    pub fn kwargs_for(&self, gear: &str) -> Option<&Map<String, Value>> {
        self.kwargs.get(gear)
    }
}

/// The gears to advertise for one checkpoint, from its probed profile.
///
/// The build order is the whole content of this function, and it is
/// least-thinking-first: `off` if the template can be turned off,
/// `adaptive` if it has that third state, then the effort ladder
/// ascending by scale -- or a bare `on` when there is a toggle but no
/// ladder, since `off` needs a counterpart to be worth offering.
///
/// `parser_configured` covers the one case the template cannot speak
/// for: an always-thinking family whose template has no observable knob
/// at all, but which really does reason and whose output really is
/// being split. It advertises a single `on` gear with EMPTY kwargs --
/// there is nothing to send, and the point is only to let a client
/// label the state rather than show a reasoning model as having none.
///
/// A template with no toggle, no adaptive state, no graded effort and
/// no parser gets an EMPTY list. It is not a reasoning model with zero
/// gears; it is a model that was asked and had nothing to say, and a
/// client should see no such field rather than an empty one.
///
/// The default is what the bare render already matches -- `off` or
/// `adaptive` when the probe found the template sitting in one of them
/// -- and otherwise the probed effort default. Failing both, `medium`
/// if it is on offer (the ecosystem's neutral gear) and otherwise the
/// last gear, because a ladder with no stated default is one whose top
/// is its normal operating point.
pub fn derive_think_gears(profile: &ThinkingProfile, parser_configured: bool) -> ThinkGears {
    let mut supported: Vec<String> = Vec::new();
    let mut kwargs: BTreeMap<String, Map<String, Value>> = BTreeMap::new();
    let mut offer = |gear: &str, k: Map<String, Value>, list: &mut Vec<String>| {
        list.push(gear.to_string());
        kwargs.insert(gear.to_string(), k);
    };

    if profile.toggleable {
        offer("off", thinking_off_kwargs(), &mut supported);
    }
    if profile.has_adaptive {
        offer("adaptive", thinking_adaptive_kwargs(), &mut supported);
    }

    let efforts = effective_efforts(&profile.efforts);
    if !efforts.is_empty() {
        for effort in efforts.iter().copied() {
            // The toggle and the gear are separate knobs: a template
            // that can be turned off needs to be told it is ON as well
            // as how hard, or selecting a gear on a defaults-off
            // checkpoint renders with no reasoning block at all.
            let mut k = if profile.toggleable {
                thinking_on_kwargs()
            } else {
                Map::new()
            };
            k.insert("reasoning_effort".into(), json!(effort.as_str()));
            offer(effort.as_str(), k, &mut supported);
        }
    } else if profile.toggleable {
        offer("on", thinking_on_kwargs(), &mut supported);
    } else if parser_configured {
        offer("on", Map::new(), &mut supported);
    }

    let has = |gear: &str| supported.iter().any(|g| g == gear);
    let default = match profile.default_state {
        ThinkingState::Off if has("off") => Some("off".to_string()),
        ThinkingState::Adaptive if has("adaptive") => Some("adaptive".to_string()),
        _ => None,
    }
    .or_else(|| {
        if efforts.is_empty() {
            return None;
        }
        profile
            .efforts
            .default
            .map(|e| e.as_str().to_string())
            .filter(|e| has(e))
            .or_else(|| has("medium").then(|| "medium".to_string()))
    })
    .or_else(|| has("on").then(|| "on".to_string()))
    .or_else(|| supported.last().cloned());

    ThinkGears {
        default: default.filter(|_| !supported.is_empty()),
        supported,
        kwargs,
    }
}

/// Broadcast the effort in every spelling the ecosystem's templates
/// read, the same rule the thinking toggles use: a graded-strength
/// template reads `reasoning_strength`, and a Jinja template ignores
/// variables it does not declare. An explicit caller-provided spelling
/// wins over the broadcast.
pub fn broadcast_effort_spellings(chat_template_kwargs: &mut Map<String, Value>) {
    if let Some(effort) = chat_template_kwargs.get("reasoning_effort").cloned() {
        chat_template_kwargs
            .entry("reasoning_strength".to_string())
            .or_insert(effort);
    }
}

fn set_match(table: &mut [(Effort, bool)], effort: Effort, value: bool) {
    if let Some(slot) = table.iter_mut().find(|(e, _)| *e == effort) {
        slot.1 = value;
    }
}

fn matches(table: &[(Effort, bool)], effort: Effort) -> bool {
    table
        .iter()
        .find(|(e, _)| *e == effort)
        .map(|(_, v)| *v)
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(supported: &[Effort], default: Option<Effort>) -> EffortProfile {
        EffortProfile {
            supported: supported.iter().copied().collect(),
            default,
            consumes_effort: true,
            validates: true,
            strength_dialect: false,
        }
    }

    #[test]
    fn in_vocabulary_values_pass_through() {
        let p = profile(&[Effort::Low, Effort::Medium, Effort::High], None);
        assert_eq!(quantize_effort("medium", &p), Some(Effort::Medium));
    }

    #[test]
    fn unknown_names_and_inert_templates_send_nothing() {
        let p = profile(&[Effort::Low, Effort::High], None);
        assert_eq!(quantize_effort("turbo", &p), None);
        assert_eq!(quantize_effort("high", &EffortProfile::inert()), None);
    }

    /// The shared rule for a two-gear encoder: `medium` (0.7) is 0.2 from
    /// `high` (0.9), past the quantize distance, so it drops to the
    /// template default rather than silently maxing the model out.
    #[test]
    fn medium_does_not_escalate_to_a_far_high() {
        let p = profile(&[Effort::None, Effort::High], None);
        assert_eq!(quantize_effort("medium", &p), None);
    }

    /// ... while `high` (0.9) is only 0.09 from `xhigh` (0.99), so it
    /// does reach it.
    #[test]
    fn high_reaches_xhigh() {
        let p = profile(&[Effort::Low, Effort::XHigh], None);
        assert_eq!(quantize_effort("high", &p), Some(Effort::XHigh));
    }

    /// `max` shares `xhigh`'s position, so rounding would always be
    /// able to reach it. It must be reachable only by its own name.
    #[test]
    fn max_is_never_entered_by_rounding() {
        let p = profile(&[Effort::Low, Effort::Max], None);
        assert_eq!(quantize_effort("high", &p), None);
        assert_eq!(quantize_effort("max", &p), Some(Effort::Max));
    }

    /// A template that accepted the whole scale validated nothing, so
    /// it is capped to the OpenAI triple instead of interpolating
    /// `minimal` verbatim.
    #[test]
    fn unvalidated_grader_is_capped_to_its_ladder() {
        let all: BTreeSet<Effort> = KNOWN_REASONING_EFFORTS.iter().copied().collect();
        let p = EffortProfile {
            supported: all.clone(),
            default: None,
            consumes_effort: true,
            validates: false,
            strength_dialect: false,
        };
        assert_eq!(
            effective_efforts(&p),
            OPENAI_EFFORT_TRIPLE.iter().copied().collect()
        );
        assert_eq!(quantize_effort("minimal", &p), Some(Effort::Low));

        let strength = EffortProfile {
            strength_dialect: true,
            ..p
        };
        assert_eq!(
            effective_efforts(&strength),
            GRADED_EFFORT_LADDER.iter().copied().collect()
        );
        // The cap is on what reaches the *template*: `max` is not in
        // this dialect's ladder, so it quantizes onto `xhigh` (which
        // shares its position) rather than being interpolated
        // verbatim. Rounding never enters a *supported* `max` -- that
        // is the separate rule below.
        assert_eq!(quantize_effort("max", &strength), Some(Effort::XHigh));
    }

    /// The probed default stays in the served vocabulary even when the
    /// dialect ladder would have dropped it.
    #[test]
    fn probed_default_is_always_offered() {
        let all: BTreeSet<Effort> = KNOWN_REASONING_EFFORTS.iter().copied().collect();
        let p = EffortProfile {
            supported: all,
            default: Some(Effort::None),
            consumes_effort: true,
            validates: false,
            strength_dialect: false,
        };
        assert!(effective_efforts(&p).contains(&Effort::None));
    }

    /// A template whose rendering never moves and never raises accepts
    /// every gear in the sense that nothing errored -- but it *reads*
    /// none of them, so the served vocabulary is empty and requests
    /// carry no effort at all.
    #[test]
    fn a_template_that_ignores_effort_is_served_no_vocabulary() {
        let p = probe_effort_profile(|_kwargs, _tools| Ok::<_, ()>("same".to_string()));
        assert!(!p.consumes_effort);
        assert!(effective_efforts(&p).is_empty());
        assert_eq!(quantize_effort("high", &p), None);
    }

    /// A template that raises on everything -- including the no-effort
    /// baseline -- teaches nothing, so nothing is sent.
    #[test]
    fn a_template_that_rejects_every_shape_is_inert() {
        let p = probe_effort_profile(|_kwargs, _tools| Err::<String, _>(()));
        assert_eq!(p, EffortProfile::inert());
    }

    /// A validating template: it renders three gears and raises on the
    /// rest, so `supported` is a real vocabulary.
    #[test]
    fn probe_learns_a_validated_vocabulary() {
        let p = probe_effort_profile(|kwargs, _tools| {
            match kwargs.get("reasoning_effort").and_then(|v| v.as_str()) {
                None => Ok("base".to_string()),
                Some(name) if OPENAI_EFFORT_TRIPLE.iter().any(|e| e.as_str() == name) => {
                    Ok(format!("base+{name}"))
                }
                Some(_) => Err(()),
            }
        });
        assert!(p.consumes_effort);
        assert!(p.validates);
        assert_eq!(
            p.supported,
            OPENAI_EFFORT_TRIPLE
                .iter()
                .copied()
                .collect::<BTreeSet<_>>()
        );
        // Every accepted gear moved the rendering, so none of them is
        // the bare-render default.
        assert_eq!(p.default, None);
    }

    /// A gear whose rendering is byte-identical to the bare render is
    /// the template's own default, and the highest such gear wins.
    #[test]
    fn probe_finds_the_highest_baseline_matching_default() {
        let p = probe_effort_profile(|kwargs, _tools| {
            match kwargs.get("reasoning_effort").and_then(|v| v.as_str()) {
                Some("low") | Some("medium") => Ok("base".to_string()),
                Some("high") => Ok("hard".to_string()),
                Some(_) => Err(()),
                None => Ok("base".to_string()),
            }
        });
        assert_eq!(p.default, Some(Effort::Medium));
    }

    #[test]
    fn thinking_probe_reads_the_toggle_and_its_default() {
        let p = probe_thinking_profile(
            |kwargs, _tools| {
                let enabled = kwargs.get("enable_thinking").and_then(|v| v.as_bool());
                let mode = kwargs.get("thinking_mode").and_then(|v| v.as_str());
                Ok::<_, ()>(match (enabled, mode) {
                    (Some(false), _) => "off".to_string(),
                    (Some(true), _) => "on".to_string(),
                    (None, Some("adaptive")) => "adaptive".to_string(),
                    _ => "off".to_string(),
                })
            },
            EffortProfile::inert(),
        );
        assert!(p.toggleable);
        assert!(p.has_adaptive);
        assert_eq!(p.default_state, ThinkingState::Off);
    }

    /// Tools force thinking: an encoder that only emits well-formed
    /// tool calls inside a reasoning block would otherwise be asked for
    /// tool calls it cannot produce.
    #[test]
    fn tools_or_an_explicit_opt_in_turn_thinking_on() {
        let tools = probe_tools();
        assert_eq!(
            resolve_thinking_mode(None, Some(&tools)),
            ThinkingMode::Thinking
        );
        let mut ctk = Map::new();
        ctk.insert("enable_thinking".into(), json!(true));
        assert_eq!(
            resolve_thinking_mode(Some(&ctk), None),
            ThinkingMode::Thinking
        );
        assert_eq!(resolve_thinking_mode(None, None), ThinkingMode::Chat);
        assert_eq!(resolve_thinking_mode(None, Some(&[])), ThinkingMode::Chat);
    }

    /// A `thinking_mode` this engine does not know must not reach the
    /// template.
    #[test]
    fn an_unknown_thinking_mode_falls_back_to_chat() {
        let mut ctk = Map::new();
        ctk.insert("thinking_mode".into(), json!("ultra"));
        assert_eq!(resolve_thinking_mode(Some(&ctk), None), ThinkingMode::Chat);
        ctk.insert("thinking_mode".into(), json!("thinking"));
        assert_eq!(
            resolve_thinking_mode(Some(&ctk), None),
            ThinkingMode::Thinking
        );
    }

    #[test]
    fn sanitizing_quantizes_in_place_and_drops_what_cannot_map() {
        let p = profile(&[Effort::Low, Effort::XHigh], None);
        let mut ctk = Map::new();
        ctk.insert("reasoning_effort".into(), json!("high"));
        assert_eq!(
            sanitize_effort(&mut ctk, &p),
            EffortMapping::Mapped(Effort::XHigh)
        );
        assert_eq!(ctk["reasoning_effort"], json!("xhigh"));

        ctk.insert("reasoning_effort".into(), json!("low"));
        assert_eq!(sanitize_effort(&mut ctk, &p), EffortMapping::Unchanged);

        ctk.insert("reasoning_effort".into(), json!("medium"));
        assert_eq!(sanitize_effort(&mut ctk, &p), EffortMapping::Dropped);
        assert!(!ctk.contains_key("reasoning_effort"));

        // A request that carried no effort is untouched.
        let mut bare = Map::new();
        assert_eq!(sanitize_effort(&mut bare, &p), EffortMapping::Unchanged);
        assert!(bare.is_empty());
    }

    #[test]
    fn the_effort_is_broadcast_but_an_explicit_spelling_wins() {
        let mut ctk = Map::new();
        ctk.insert("reasoning_effort".into(), json!("high"));
        broadcast_effort_spellings(&mut ctk);
        assert_eq!(ctk["reasoning_strength"], json!("high"));

        let mut explicit = Map::new();
        explicit.insert("reasoning_effort".into(), json!("high"));
        explicit.insert("reasoning_strength".into(), json!("low"));
        broadcast_effort_spellings(&mut explicit);
        assert_eq!(explicit["reasoning_strength"], json!("low"));
    }

    #[test]
    fn a_template_with_no_toggle_defaults_to_on() {
        let p = probe_thinking_profile(
            |_kwargs, _tools| Ok::<_, ()>("same".to_string()),
            EffortProfile::inert(),
        );
        assert!(!p.toggleable);
        assert_eq!(p.default_state, ThinkingState::On);
    }

    fn thinking_profile(
        toggleable: bool,
        has_adaptive: bool,
        default_state: ThinkingState,
        efforts: EffortProfile,
    ) -> ThinkingProfile {
        ThinkingProfile {
            efforts,
            toggleable,
            has_adaptive,
            default_state,
        }
    }

    /// The build order is the content of the function, so the test is
    /// an equality on the whole list rather than a membership check.
    #[test]
    fn gears_are_built_least_thinking_first() {
        let gears = derive_think_gears(
            &thinking_profile(
                true,
                true,
                ThinkingState::Off,
                profile(&[Effort::Low, Effort::Medium, Effort::High], None),
            ),
            false,
        );
        assert_eq!(
            gears.supported,
            ["off", "adaptive", "low", "medium", "high"]
        );
        assert_eq!(gears.default.as_deref(), Some("off"));
    }

    /// `on` exists only as the counterpart to `off`: a toggle with no
    /// ladder still needs something to name the other position.
    #[test]
    fn a_toggle_with_no_ladder_offers_a_bare_on() {
        let gears = derive_think_gears(
            &thinking_profile(true, false, ThinkingState::On, EffortProfile::inert()),
            false,
        );
        assert_eq!(gears.supported, ["off", "on"]);
        assert_eq!(gears.default.as_deref(), Some("on"));
    }

    /// The distinction the acceptance criterion turns on: a checkpoint
    /// that says nothing about thinking advertises NOTHING, not an
    /// empty list, which would read as "asked, and it has no gears".
    #[test]
    fn a_checkpoint_that_grades_nothing_advertises_nothing() {
        let gears = derive_think_gears(
            &thinking_profile(false, false, ThinkingState::On, EffortProfile::inert()),
            false,
        );
        assert!(gears.is_empty());
        assert_eq!(gears.default, None);
    }

    #[test]
    fn the_probed_effort_default_wins_when_the_bare_render_is_thinking() {
        let gears = derive_think_gears(
            &thinking_profile(
                false,
                false,
                ThinkingState::On,
                profile(
                    &[Effort::Low, Effort::Medium, Effort::High],
                    Some(Effort::High),
                ),
            ),
            false,
        );
        assert_eq!(gears.supported, ["low", "medium", "high"]);
        assert_eq!(gears.default.as_deref(), Some("high"));
    }

    /// No stated default: `medium` when it is on offer, and otherwise
    /// the top of the ladder.
    #[test]
    fn a_ladder_with_no_stated_default_falls_back_to_medium_then_to_its_top() {
        let with_medium = derive_think_gears(
            &thinking_profile(
                false,
                false,
                ThinkingState::On,
                profile(&[Effort::Low, Effort::Medium, Effort::High], None),
            ),
            false,
        );
        assert_eq!(with_medium.default.as_deref(), Some("medium"));

        let without = derive_think_gears(
            &thinking_profile(
                false,
                false,
                ThinkingState::On,
                profile(&[Effort::Low, Effort::XHigh], None),
            ),
            false,
        );
        assert_eq!(without.default.as_deref(), Some("xhigh"));
    }

    /// The half a client cannot derive: what to SEND to select a gear.
    /// Without it a UI still has to know that "off" is two booleans and
    /// "high" is a string, which is the family knowledge this module
    /// exists to remove.
    #[test]
    fn every_advertised_gear_carries_the_kwargs_that_select_it() {
        let gears = derive_think_gears(
            &thinking_profile(
                true,
                true,
                ThinkingState::Off,
                profile(&[Effort::Low, Effort::High], None),
            ),
            false,
        );
        assert_eq!(gears.supported, ["off", "adaptive", "low", "high"]);
        for gear in &gears.supported {
            assert!(gears.kwargs_for(gear).is_some(), "{gear} has no kwargs");
        }
        assert_eq!(
            gears.kwargs_for("off").unwrap()["enable_thinking"],
            json!(false)
        );
        // A gear on a toggleable template must also say "and turn it
        // on": selecting `high` on a defaults-off checkpoint otherwise
        // renders with no reasoning block at all.
        let high = gears.kwargs_for("high").unwrap();
        assert_eq!(high["reasoning_effort"], json!("high"));
        assert_eq!(high["enable_thinking"], json!(true));
    }

    /// A template with no toggle does not need to be told it is on --
    /// it always is -- so the gear carries the effort alone.
    #[test]
    fn a_gear_on_an_untoggleable_template_carries_only_the_effort() {
        let gears = derive_think_gears(
            &thinking_profile(
                false,
                false,
                ThinkingState::On,
                profile(&[Effort::Low, Effort::High], None),
            ),
            false,
        );
        let low = gears.kwargs_for("low").unwrap();
        assert_eq!(low.len(), 1);
        assert_eq!(low["reasoning_effort"], json!("low"));
    }

    /// The case the template cannot speak for. An always-thinking
    /// family really does reason and its output really is being split;
    /// advertising nothing would show it as a model with no gears.
    #[test]
    fn an_always_thinking_family_with_no_knob_still_advertises_its_state() {
        let profile = thinking_profile(false, false, ThinkingState::On, EffortProfile::inert());

        let unparsed = derive_think_gears(&profile, false);
        assert!(unparsed.is_empty(), "nothing reasons and nothing is parsed");

        let parsed = derive_think_gears(&profile, true);
        assert_eq!(parsed.supported, ["on"]);
        assert_eq!(parsed.default.as_deref(), Some("on"));
        // Empty, deliberately: there is nothing to send. The gear
        // exists so a client can LABEL the state, not select it.
        assert!(parsed.kwargs_for("on").unwrap().is_empty());
    }

    /// A default that names a gear outside the advertised list would
    /// be unaskable; the adaptive state is only the default when it is
    /// actually offered.
    #[test]
    fn the_default_is_always_a_member_of_the_advertised_list() {
        let gears = derive_think_gears(
            &thinking_profile(
                false,
                false,
                ThinkingState::Adaptive,
                profile(&[Effort::Low, Effort::Medium], None),
            ),
            false,
        );
        assert!(!gears.supported.contains(&"adaptive".to_string()));
        assert_eq!(gears.default.as_deref(), Some("medium"));
        assert!(gears
            .default
            .as_ref()
            .is_some_and(|d| gears.supported.contains(d)));
    }
}
