//! The variables every llama.cpp render carries beside a request's own
//! (`common/chat.cpp`: `common_chat_templates_apply_jinja` and
//! `common_chat_template_direct_apply_impl`). Upstream sets each before
//! the kwargs, so a kwarg of the same name wins, here as there.
//!
//! | Variable | Value |
//! |---|---|
//! | `enable_thinking` | [`RenderOptions::enable_thinking`], which llama.cpp sets from whether the template supports thinking ([`ChatTemplate::supports_thinking`]) |
//! | `date_string`, `datetime` | today in local time, `%d %b %Y` and `%b %d %Y` (`common_chat_extra_context`) |
//! | `reasoning_strength` | a non-empty `reasoning_effort` (`caps_apply_reasoning_effort`) |
//! | `preserve_thinking`, and its opposite as `clear_thinking`, `truncate_history_thinking` and `drop_thinking` | a boolean `preserve_reasoning` (`caps_apply_preserve_reasoning`) |
//!
//! The clock is local time, as llama.cpp's `std::localtime`, for these
//! dates and for `strftime_now` alike.

use serde_json::{json, Map, Value};

use super::{format_utc, ChatTemplate, RenderOptions};

/// What llama.cpp's own analysis sends as the user's turn when it probes
/// a template (`common/chat-diff-analyzer.cpp`).
const PROBE_USER: &str = "U_USER_MSG Hello END_U";

/// Fills in each variable the kwargs already in `ctx` did not set.
pub(super) fn fill(
    ctx: &mut Map<String, Value>,
    enable_thinking: Option<bool>,
) -> Result<(), String> {
    let effort = ctx.get("reasoning_effort").cloned();
    if let Some(effort) = effort.filter(|e| e.as_str().is_some_and(|s| !s.is_empty())) {
        ctx.entry("reasoning_strength").or_insert(effort);
    }
    if let Some(&Value::Bool(keep)) = ctx.get("preserve_reasoning") {
        for (name, value) in preserve_reasoning_spellings(keep) {
            ctx.entry(name).or_insert(Value::Bool(value));
        }
    }
    if let Some(on) = enable_thinking {
        ctx.entry("enable_thinking").or_insert(Value::Bool(on));
    }
    let now = local_now()?;
    for (name, format) in [("date_string", "%d %b %Y"), ("datetime", "%b %d %Y")] {
        if !ctx.contains_key(name) {
            let date = format_utc(now, format).map_err(|spec| format!("{name}: %{spec}"))?;
            ctx.insert(name.to_string(), Value::String(date));
        }
    }
    Ok(())
}

/// `caps_apply_preserve_reasoning`: GLM reads `clear_thinking`, LFM
/// `preserve_thinking`, Nemotron `truncate_history_thinking`, DeepSeek-V4
/// `drop_thinking`.
pub(super) fn preserve_reasoning_spellings(keep: bool) -> [(&'static str, bool); 4] {
    [
        ("preserve_thinking", keep),
        ("clear_thinking", !keep),
        ("truncate_history_thinking", !keep),
        ("drop_thinking", !keep),
    ]
}

/// Now, in local time, as the Unix seconds [`format_utc`] prints: the
/// instant plus this zone's offset from UTC at it.
/// `FRINK_TEST_CHAT_TEMPLATE_NOW` pins it to those seconds, read as UTC.
pub(super) fn local_now() -> Result<i64, String> {
    #[cfg(test)]
    if let Some(secs) = PINNED.with(std::cell::Cell::get) {
        return Ok(secs);
    }
    if let Ok(pinned) = std::env::var("FRINK_TEST_CHAT_TEMPLATE_NOW") {
        return pinned
            .trim()
            .parse()
            .map_err(|_| "FRINK_TEST_CHAT_TEMPLATE_NOW must be Unix seconds".to_string());
    }
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    Ok(local_seconds(secs))
}

/// `secs` shifted to this zone's wall clock, for [`format_utc`] to print.
fn local_seconds(secs: i64) -> i64 {
    secs + utc_offset(secs)
}

/// This zone's offset from UTC at `secs`, in seconds, as `localtime`
/// gives it. Zero where there is no `localtime_r`.
#[cfg(unix)]
// `tm_gmtoff` is a `c_long`: `i64` here, `i32` on a 32-bit target.
#[allow(clippy::useless_conversion)]
fn utc_offset(secs: i64) -> i64 {
    local_tm(secs).map_or(0, |tm| i64::from(tm.tm_gmtoff))
}

#[cfg(not(unix))]
fn utc_offset(_secs: i64) -> i64 {
    0
}

#[cfg(unix)]
fn local_tm(secs: i64) -> Option<libc::tm> {
    let time = libc::time_t::try_from(secs).ok()?;
    // SAFETY: `tm` is integers and one nullable pointer, so all zeroes is
    // a valid value of it.
    let mut tm: libc::tm = unsafe { std::mem::zeroed() };
    // SAFETY: `localtime_r` reads `time` and writes one `tm` through a
    // pointer to storage this frame owns, keeping neither. Null means it
    // failed, and then `tm` is not read.
    let filled = unsafe { libc::localtime_r(&time, &mut tm) };
    (!filled.is_null()).then_some(tm)
}

impl ChatTemplate {
    /// Whether llama.cpp calls this template thinking-capable
    /// (`common_chat_templates_support_enable_thinking`), which is what
    /// `enable_thinking` defaults to in every llama-server and llama-cli
    /// render.
    ///
    /// Upstream answers from a per-family handler or its template analysis
    /// (`common/chat-diff-analyzer.cpp`, `compare_thinking_enabled`). This
    /// asks the question that analysis starts from, with its probe
    /// message: does turning thinking on change what follows the user's
    /// turn? A template whose switch only rewrites its system block
    /// (Apertus's "Deliberation: enabled") does not count, as upstream does
    /// not count it. One that ignores the variable gets `false`, which it
    /// ignores. It is a simpler rule than upstream's, and on every template
    /// llama.cpp bundles it gives the prompt llama.cpp does, which
    /// frink-server's render census checks.
    pub fn supports_thinking(&self, bos_token: Option<&str>, eos_token: Option<&str>) -> bool {
        let render = |on: bool| {
            let mut extra = Map::new();
            extra.insert("enable_thinking".to_string(), Value::Bool(on));
            self.render(
                &[json!({"role": "user", "content": PROBE_USER})],
                &RenderOptions {
                    add_generation_prompt: true,
                    bos_token: bos_token.map(str::to_string),
                    eos_token: eos_token.map(str::to_string),
                    extra,
                    ..Default::default()
                },
            )
        };
        let (Ok(off), Ok(on)) = (render(false), render(true)) else {
            return false;
        };
        let shared = off
            .bytes()
            .rev()
            .zip(on.bytes().rev())
            .take_while(|(a, b)| a == b)
            .count();
        let tail = &off.as_bytes()[off.len() - shared..];
        off != on
            && !tail
                .windows(PROBE_USER.len())
                .any(|w| w == PROBE_USER.as_bytes())
    }
}

#[cfg(test)]
thread_local! {
    static PINNED: std::cell::Cell<Option<i64>> = const { std::cell::Cell::new(None) };
}

/// Pins this thread's clock to `secs` (read as UTC) until it drops, so a
/// test asserts a date without touching the process's environment.
#[cfg(test)]
pub(super) struct PinnedClock(Option<i64>);

#[cfg(test)]
impl PinnedClock {
    pub(super) fn at(secs: i64) -> Self {
        Self(PINNED.with(|p| p.replace(Some(secs))))
    }
}

#[cfg(test)]
impl Drop for PinnedClock {
    fn drop(&mut self) {
        PINNED.with(|p| p.set(self.0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template(source: &str) -> ChatTemplate {
        ChatTemplate::from_jinja(source).unwrap()
    }

    fn render(source: &str, extra: Value, enable_thinking: Option<bool>) -> String {
        let Value::Object(extra) = extra else {
            panic!("not an object")
        };
        template(source)
            .render(
                &[json!({"role": "user", "content": "x"})],
                &RenderOptions {
                    extra,
                    enable_thinking,
                    ..Default::default()
                },
            )
            .unwrap()
    }

    /// The two date variables, in upstream's two formats, under a kwarg.
    #[test]
    fn dates_are_today_under_a_kwarg() {
        let _clock = PinnedClock::at(1_720_096_496); // 2024-07-04T12:34:56Z
        let source = "{{ date_string }}|{{ datetime }}";
        assert_eq!(render(source, json!({}), None), "04 Jul 2024|Jul 04 2024");
        assert_eq!(
            render(source, json!({"date_string": "26 Jul 2024"}), None),
            "26 Jul 2024|Jul 04 2024"
        );
    }

    /// llama.cpp passes `enable_thinking` to every render, and a kwarg
    /// still says otherwise.
    #[test]
    fn enable_thinking_is_the_default_under_a_kwarg() {
        let source = "{{ enable_thinking }}";
        assert_eq!(render(source, json!({}), Some(true)), "True");
        assert_eq!(render(source, json!({}), Some(false)), "False");
        assert_eq!(
            render(source, json!({"enable_thinking": false}), Some(true)),
            "False"
        );
        assert_eq!(render(source, json!({}), None), "");
    }

    /// On at the end of the prompt counts, on in the system block alone
    /// does not, and a template that ignores the switch is not one.
    #[test]
    fn supports_thinking_asks_what_follows_the_users_turn() {
        let qwen3 = "{% for m in messages %}<|user|>{{ m.content }}{% endfor %}<|assistant|>\
                     {% if enable_thinking is false %}<think></think>{% endif %}";
        let apertus = "<|system|>Deliberation: {{ 'enabled' if enable_thinking else 'disabled' }}\
                       {% for m in messages %}<|user|>{{ m.content }}{% endfor %}<|assistant|>";
        let gemma4 = "{% if enable_thinking %}<|turn>system\n<|think|><turn|>{% endif %}\
                      {% for m in messages %}<|turn>user\n{{ m.content }}<turn|>{% endfor %}<|turn>model\n\
                      {% if not enable_thinking %}<|channel>thought<channel|>{% endif %}";
        let plain = "{% for m in messages %}<|user|>{{ m.content }}{% endfor %}<|assistant|>";
        assert!(template(qwen3).supports_thinking(None, None));
        assert!(!template(apertus).supports_thinking(None, None));
        assert!(template(gemma4).supports_thinking(None, None));
        assert!(!template(plain).supports_thinking(None, None));
        assert!(!template("{{ raise_exception('no') }}").supports_thinking(None, None));
    }

    /// The offset is `localtime_r`'s, and the pure-Rust calendar printing
    /// the shifted seconds gives the fields `localtime_r` gives, in
    /// whatever zone this runs.
    #[cfg(unix)]
    #[test]
    fn local_time_is_localtime_r() {
        for secs in [0, 951_782_400, 1_720_096_496, 1_794_118_492, 4_102_444_800] {
            let tm = local_tm(secs).expect("localtime_r");
            let ours = format_utc(local_seconds(secs), "%Y %m %d %H %M %S %j").unwrap();
            let theirs = format!(
                "{} {:02} {:02} {:02} {:02} {:02} {:03}",
                tm.tm_year + 1900,
                tm.tm_mon + 1,
                tm.tm_mday,
                tm.tm_hour,
                tm.tm_min,
                tm.tm_sec,
                tm.tm_yday + 1
            );
            assert_eq!(ours, theirs, "{secs}");
        }
    }
}
