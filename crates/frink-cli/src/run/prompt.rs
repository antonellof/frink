//! The prompt `frink run` feeds the model: `-p`/`-f`, escapes, and the
//! chat template. Moved out of `run/mod.rs`, with `final_prompt` as the
//! one place every generation path builds it.

use super::*;

/// The checkpoint's own chat template, evaluated.
///
/// This used to be a near-identical copy of `frink-server`'s marker
/// sniffer -- six hand-written renderers picked by which literal marker
/// a template string happened to contain. Both are gone; both now
/// compile the real Jinja source with
/// [`frink_models::chat_template`], so `frink -m mistral.gguf -p hi`
/// and `POST /v1/chat/completions` frame the same conversation the same
/// way, including for the families the sniffer never recognised.
pub(super) struct ChatKind {
    template: frink_models::chat_template::ChatTemplate,
    bos_token: Option<String>,
    eos_token: Option<String>,
}

impl ChatKind {
    pub(super) fn detect_for_gguf(file: &ShardedGguf, byte_tokenizer: bool) -> Self {
        ChatKind {
            template: frink_models::chat_template::ChatTemplate::from_gguf_metadata(
                file.metadata_str("tokenizer.chat_template"),
                file.metadata_str("general.architecture"),
                byte_tokenizer,
                frink_models::chat_template::ChatTemplate::vocab_has_chatml(file),
            ),
            bos_token: file.token_text("tokenizer.ggml.bos_token_id"),
            eos_token: file.token_text("tokenizer.ggml.eos_token_id"),
        }
    }

    /// One conversation turn, framed the way this checkpoint expects.
    ///
    /// A template that will not render is an error, never a fallback to
    /// a guessed framing: a silently mis-framed prompt is exactly the
    /// bug that made this stop sniffing, and it shows up as degenerate
    /// output rather than as a message.
    pub(super) fn wrap_user(&self, system: Option<&str>, user: &str) -> anyhow::Result<String> {
        let mut messages = Vec::new();
        if let Some(sys) = system {
            messages.push(serde_json::json!({"role": "system", "content": sys}));
        }
        messages.push(serde_json::json!({"role": "user", "content": user}));
        let opts = frink_models::chat_template::RenderOptions {
            add_generation_prompt: true,
            bos_token: self.bos_token.clone(),
            eos_token: self.eos_token.clone(),
            ..Default::default()
        };
        self.template
            .render(&messages, &opts)
            .map_err(|e| anyhow::anyhow!("chat template failed to render: {e}"))
    }
}

pub(super) fn apply_escapes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// The prompt every generation path feeds the model: `-p`/`-f`, wrapped
/// in the checkpoint's chat template unless `--no-cnv`, echoed when
/// `--verbose-prompt` asks.
///
/// ONE function for the four paths. Each used to build it inline, and
/// `--verbose-prompt` reached only the generic decoder's copy: on an
/// MLA, Gemma-4 or GLM-5.2 checkpoint the flag was accepted and did
/// nothing.
pub(super) fn final_prompt(args: &InferArgs, file: &ShardedGguf) -> anyhow::Result<String> {
    let user_prompt = resolve_prompt(args)?;
    let prompt = if args.no_cnv {
        // `--system` lives inside the chat template, so a raw prompt
        // has nowhere to put it. Accepting both and dropping one is
        // the silent kind of wrong.
        if args.system.is_some() {
            anyhow::bail!(
                "--system has no effect with --no-cnv: the system prompt is placed by the chat \
                 template, which --no-cnv skips. Drop one of the two"
            );
        }
        user_prompt
    } else {
        ChatKind::detect_for_gguf(file, false).wrap_user(args.system.as_deref(), &user_prompt)?
    };
    if args.verbose_prompt {
        eprintln!("----- prompt -----");
        eprintln!("{prompt}");
        eprintln!("------------------");
    }
    Ok(prompt)
}

pub(super) fn resolve_prompt(args: &InferArgs) -> anyhow::Result<String> {
    let mut prompt = if let Some(path) = &args.file {
        let mut buf = String::new();
        let mut f = std::fs::File::open(path)?;
        f.read_to_string(&mut buf)?;
        buf
    } else {
        args.prompt.clone()
    };
    if args.escape && !args.no_escape {
        prompt = apply_escapes(&prompt);
    }
    Ok(prompt)
}
