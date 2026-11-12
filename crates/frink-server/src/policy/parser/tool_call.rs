//! Recognizing a tool call in whichever wire format a checkpoint was
//! trained on.
//!
//! There is no standard. Every model family invented its own way of
//! saying "call this function with these arguments", and a served model
//! only ever emits its own:
//!
//! | family | shape |
//! |---|---|
//! | Hermes / Qwen2.5 | `<tool_call>{"name": …, "arguments": {…}}</tool_call>` |
//! | Llama 3 | `<\|python_tag\|>{"name": …, "parameters": {…}}` |
//! | Mistral | `[TOOL_CALLS] [{…}, {…}]` |
//! | Qwen3-Coder | `<tool_call><function=name><parameter=key>value</parameter></function></tool_call>` |
//! | GLM-4.7 | `<tool_call>name\n<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` |
//! | DeepSeek | `<｜DSML｜invoke name="…"><｜DSML｜parameter name="…">…` |
//! | MiniMax-M2 | `<minimax:tool_call><invoke name="…"><parameter name="…">…` |
//! | MiniMax-M3 | `]<]minimax[>[<tool_call>]<]minimax[>[<invoke name="…">]<]minimax[>[<key>…` |
//! | gpt-oss | `<\|channel\|>commentary to=functions.name<\|message\|>{…}<\|call\|>` |
//! | Gemma 4 | `<\|tool_call>call:name{k: v}<tool_call\|>` |
//! | muse-glimmer | `<\|start\|>assistant to=tool<\|message\|><atem:function_calls>…<\|eot\|>` |
//!
//! Two of these are *not* variations on the others, and both were
//! mis-parsed by being treated as one:
//!
//! **MiniMax-M3 is a different protocol from M2's**, not M2 with
//! renamed tags. Every structural tag carries the `]<]minimax[>[`
//! namespace prefix, one wrapper holds several `<invoke>` tags, and the
//! arguments are parameter-*name* elements rendered recursively --
//! objects nest tags, arrays repeat `<item>` -- typed from the tool's
//! schema at every nesting level.
//!
//! **muse-glimmer wraps its calls in a channel**. The invoke/parameter
//! block is ordinary, but it only *counts* inside a channel addressed
//! to a tool ([`ToolCallFormat::MuseGlimmer`]); the same block written
//! in an answer to the user -- the system prompt's own example, echoed
//! back -- is text. Everything else the model says in a tool channel is
//! dropped, because the template puts nothing else there.
//!
//! # Why the XML-ish families need a schema
//!
//! A JSON family states its own types: `{"count": 3}` is a number
//! because it is written as one. An XML-ish family does not --
//! `<parameter name="count">3</parameter>` is a *string* on the wire,
//! and whether it should reach the tool as `3` or `"3"` is a fact about
//! the tool's schema, not about the text. So [`ToolSchema`] is passed
//! in, and a declared `integer` parameter is parsed while a declared
//! `string` one is handed over verbatim -- which is what keeps a
//! zero-padded id like `"018956"` from arriving as `18956`.
//!
//! # What streams and what does not
//!
//! The invoke/parameter families stream **prefix-stable** JSON
//! fragments: each fragment is a literal continuation of the arguments
//! JSON, so a client can concatenate them and parse the result. That
//! matters for coding agents, whose arguments are whole files.
//!
//! The JSON-payload families (Hermes, Llama 3, Mistral, gpt-oss,
//! Gemma) are emitted whole, when their block completes. Their payload
//! is not meaningfully parseable until it is complete, and emitting a
//! half-written JSON object as a "fragment" would only move the
//! problem to the client.
//!
//! Ported 1:1 from FreeToken's `server/function_call_parser.py`
//! (Apache-2.0); see `docs/THIRD_PARTY_NOTICES.md`.

use std::collections::BTreeMap;

use serde_json::{Map, Value};

use crate::policy::detokenize::{floor_char_boundary, stop_prefix_holdback};
use crate::policy::parser::reasoning::{
    atem_boundary, atem_hold_len, atem_marker_inside, atem_recipient, AtemBoundary,
    ATEM_CLOSING_TOKENS, ATEM_HEADER_SPAN, ATEM_MESSAGE, ATEM_START,
};

/// One recognized call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// Position in this response's list of calls.
    pub index: usize,
    pub name: String,
    /// The arguments as a JSON object string, ready to hand to a
    /// client. Always valid JSON, `{}` when the call took none.
    pub arguments: String,
}

/// What a streaming parse produced, in wire order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallEvent {
    /// Ordinary output, safe to send.
    Text(String),
    /// A call has begun. Its arguments follow.
    CallStart { index: usize, name: String },
    /// A literal continuation of the call's arguments JSON.
    CallArguments { index: usize, fragment: String },
    /// The call is complete, with its whole arguments object.
    CallEnd { index: usize, arguments: String },
}

/// One tool the request offered, used to type XML-ish parameter values.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ToolSchema {
    pub name: String,
    /// The `parameters` JSON Schema, if the request supplied one.
    pub parameters: Option<Value>,
}

impl ToolSchema {
    pub fn new(name: impl Into<String>) -> Self {
        ToolSchema {
            name: name.into(),
            parameters: None,
        }
    }

    pub fn with_parameters(name: impl Into<String>, parameters: Value) -> Self {
        ToolSchema {
            name: name.into(),
            parameters: Some(parameters),
        }
    }

    /// The declared JSON Schema type of one parameter, if any.
    fn parameter_type(&self, key: &str) -> Option<String> {
        let properties = self.parameters.as_ref()?.get("properties")?;
        let declared = properties.get(key)?;
        declared
            .get("type")
            .and_then(Value::as_str)
            .map(str::to_string)
    }
}

/// Which family's tool-call format to parse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// Hermes-style `<tool_call>` wrapping a JSON object.
    Qwen25,
    /// `<|python_tag|>` followed by one or more JSON objects.
    Llama3,
    /// `[TOOL_CALLS] [ … ]`.
    Mistral,
    /// `<function=`/`<parameter=` inside `<tool_call>`.
    Qwen3Coder,
    /// `<arg_key>`/`<arg_value>` inside `<tool_call>`.
    Glm47,
    /// DeepSeek's DSML invoke/parameter grammar.
    DeepSeekV32,
    /// MiniMax-M2's `<minimax:tool_call>` invoke/parameter grammar.
    MiniMax,
    /// MiniMax-M3's `]<]minimax[>[`-namespaced recursive element
    /// grammar -- a different protocol, not renamed tags.
    MiniMaxM3,
    /// The gpt-oss harmony commentary channel.
    GptOss,
    /// Gemma 4's `call:name{…}` form.
    Gemma4,
    /// muse-glimmer's ATEM channel grammar around an
    /// `<atem:invoke>`/`<atem:parameter>` block.
    MuseGlimmer,
}

/// Every marker that means "a tool call may be starting", across all
/// families. Used to decide, cheaply, whether a response is worth
/// parsing at all.
pub const TOOL_MARKERS: [&str; 12] = [
    "<tool_call>",
    "<function=",
    "<|python_tag|>",
    "[TOOL_CALLS]",
    "<minimax:tool_call>",
    "]<]minimax[>[<tool_call>",
    "<｜DSML｜function_calls>",
    "<｜DSML｜invoke",
    "<|channel|>",
    "<|tool_call>",
    "to=functions.",
    "<atem:function_calls>",
];

/// gpt-oss's harmony framing, which is not a marker pair and so has no
/// [`Markers`] entry: a call is a CHANNEL whose header is addressed to a
/// function, and `<|channel|>` on its own opens an ordinary message.
///
/// These are the literals [`ToolCallParser::parse_harmony`] reads a call
/// with, and they are `pub(crate)` because
/// [`crate::tool_grammar`] WRITES a forced call from the same ones.
/// Spelling a marker twice, once to read and once to write, is how the
/// two halves drift.
pub(crate) mod harmony {
    /// Opens a channel header.
    pub(crate) const CHANNEL_OPEN: &str = "<|channel|>";
    /// Ends the header and opens the message body.
    pub(crate) const MESSAGE_OPEN: &str = "<|message|>";
    /// Ends a message body that is a tool call.
    pub(crate) const CALL_CLOSE: &str = "<|call|>";
    /// Every marker that ends a message body. `<|start|>` is here
    /// because a model that forgets to close one starts the next.
    pub(crate) const BODY_ENDS: [&str; 4] = ["<|end|>", "<|return|>", CALL_CLOSE, "<|start|>"];
    /// The header token that addresses a message, and the namespace a
    /// tool lives in. A header token spelled `to=functions.NAME` is what
    /// makes a channel a call.
    pub(crate) const RECIPIENT_KEY: &str = "to=";
    pub(crate) const FUNCTION_NAMESPACE: &str = "functions.";
    /// The optional constrain hint the harmony spec allows in a header
    /// between the recipient and the message.
    pub(crate) const CONSTRAIN: &str = "<|constrain|>";
    /// The channels a call is written on. `parse_harmony` accepts any
    /// channel whose header addresses a function; these two are the ones
    /// the checkpoint is trained to use, per llama.cpp's
    /// `common_chat_params_init_gpt_oss`.
    pub(crate) const CHANNELS: [&str; 2] = ["commentary", "analysis"];
}

/// Gemma 4's own call syntax, which is neither JSON nor XML-ish.
///
/// A call is `<|tool_call>call:NAME{k:v,k:v}<tool_call|>`, and the
/// arguments are a comma-separated list of `key:value` pairs in gemma's
/// own quoting: a string is wrapped in [`QUOTE`], everything else is
/// written the way JSON writes it. That is the chat template's
/// `format_argument(value, escape_keys=False)`, and the pairs arrive in
/// the template's `| dictsort` order -- sorted by key.
///
/// These are the literals [`ToolCallParser::parse_gemma`] reads a call
/// with, and they are `pub(crate)` because [`crate::tool_grammar`]
/// WRITES a forced call from the same ones.
pub(crate) mod gemma {
    /// Opens and closes the call block.
    pub(crate) const BLOCK_OPEN: &str = "<|tool_call>";
    pub(crate) const BLOCK_CLOSE: &str = "<tool_call|>";
    /// Introduces the function's name inside the block.
    pub(crate) const CALL_KEY: &str = "call:";
    /// Wrap the argument list.
    pub(crate) const ARGS_OPEN: &str = "{";
    pub(crate) const ARGS_CLOSE: &str = "}";
    /// Between two pairs, and between a key and its value.
    pub(crate) const PAIR_SEPARATOR: &str = ",";
    pub(crate) const KEY_SEPARATOR: &str = ":";
    /// What a STRING value is wrapped in. Gemma spells its own quote
    /// this way; a `"` is an ordinary character inside one.
    pub(crate) const QUOTE: &str = "<|\"|>";
}

/// MiniMax-M3's namespace prefix, in front of every structural tag.
const M3_NS: &str = "]<]minimax[>[";
/// `M3_NS` plus `<`: where every M3 tag begins.
const M3_TAG: &str = "]<]minimax[>[<";
/// `M3_NS` plus `</`: where every M3 closing tag begins.
const M3_CLOSE_TAG: &str = "]<]minimax[>[</";
const M3_OPEN: &str = "]<]minimax[>[<tool_call>";
const M3_CLOSE: &str = "]<]minimax[>[</tool_call>";
const M3_INVOKE_OPEN: &str = "]<]minimax[>[<invoke";
const M3_INVOKE_CLOSE: &str = "]<]minimax[>[</invoke>";
/// Characters that cannot appear in an element's tag name. A "name"
/// holding one of these is not a parameter element at all -- it is an
/// `<invoke …>` echo or model noise -- and is stepped over.
const M3_TAG_BAD: [char; 5] = [' ', '"', '<', '>', '/'];

impl ToolCallFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            ToolCallFormat::Qwen25 => "qwen25",
            ToolCallFormat::Llama3 => "llama3",
            ToolCallFormat::Mistral => "mistral",
            ToolCallFormat::Qwen3Coder => "qwen3_coder",
            ToolCallFormat::Glm47 => "glm47",
            ToolCallFormat::DeepSeekV32 => "deepseekv32",
            ToolCallFormat::MiniMax => "minimax",
            ToolCallFormat::MiniMaxM3 => "minimax_m3",
            ToolCallFormat::GptOss => "gpt_oss",
            ToolCallFormat::Gemma4 => "gemma4",
            ToolCallFormat::MuseGlimmer => "muse_glimmer",
        }
    }

    pub fn parse(name: &str) -> Option<ToolCallFormat> {
        match name {
            "qwen" | "qwen25" => Some(ToolCallFormat::Qwen25),
            "llama3" => Some(ToolCallFormat::Llama3),
            "mistral" => Some(ToolCallFormat::Mistral),
            "qwen3_coder" => Some(ToolCallFormat::Qwen3Coder),
            "glm47" => Some(ToolCallFormat::Glm47),
            "deepseekv32" => Some(ToolCallFormat::DeepSeekV32),
            "minimax" => Some(ToolCallFormat::MiniMax),
            "minimax_m3" => Some(ToolCallFormat::MiniMaxM3),
            "gpt_oss" | "gpt-oss" => Some(ToolCallFormat::GptOss),
            "gemma4" => Some(ToolCallFormat::Gemma4),
            "muse_glimmer" => Some(ToolCallFormat::MuseGlimmer),
            _ => None,
        }
    }

    /// Which format a checkpoint's identity implies.
    ///
    /// The order of these arms is load-bearing: the specific families
    /// have to be tested before the general ones they look like, or
    /// Qwen3-Coder resolves to plain Qwen and its whole grammar is
    /// missed -- and MiniMax-M3 resolves to M2, whose `<minimax:…>`
    /// tags appear nowhere in an M3 turn, so every call it makes is
    /// emitted to the client as raw markup in `content`. Llama 3 is the
    /// fallback because its `<|python_tag|>` form is also what an
    /// untrained model most often improvises.
    pub fn infer(marker: &str) -> ToolCallFormat {
        let marker = marker.to_ascii_lowercase();
        let has = |needle: &str| marker.contains(needle);
        if has("gpt_oss") || has("gpt-oss") || has("gptoss") {
            ToolCallFormat::GptOss
        } else if has("muse") && has("glimmer") {
            ToolCallFormat::MuseGlimmer
        } else if has("minimax_m3") || has("minimax-m3") || has("minimaxm3") {
            ToolCallFormat::MiniMaxM3
        } else if has("minimax") {
            ToolCallFormat::MiniMax
        // Every spelling a real Gemma 4 file uses. `gemma4` alone was
        // the whole test, and no checkpoint is named that: the one on
        // hand calls itself `Gemma-4-E2B-It`, so this arm could not fire
        // and every Gemma 4 tool call fell through to the Llama 3
        // fallback, which reads none of them.
        } else if has("gemma4") || has("gemma-4") || has("gemma_4") || has("gemma 4") {
            ToolCallFormat::Gemma4
        } else if has("qwen3_5") || has("qwen3.5") || (has("qwen3") && has("coder")) {
            ToolCallFormat::Qwen3Coder
        } else if has("qwen") {
            ToolCallFormat::Qwen25
        } else if has("deepseek") && (has("v4") || has("v3.2") || has("v32")) {
            ToolCallFormat::DeepSeekV32
        } else if has("glm") {
            ToolCallFormat::Glm47
        } else if has("mistral") {
            ToolCallFormat::Mistral
        } else {
            ToolCallFormat::Llama3
        }
    }

    /// The marker that opens a call in this format -- what a scheduler
    /// watches for when it wants to checkpoint state at the start of a
    /// tool call.
    pub fn opener(self) -> Option<&'static str> {
        match self {
            ToolCallFormat::Qwen25 | ToolCallFormat::Qwen3Coder | ToolCallFormat::Glm47 => {
                Some("<tool_call>")
            }
            ToolCallFormat::Llama3 => Some("<|python_tag|>"),
            ToolCallFormat::Mistral => Some("[TOOL_CALLS]"),
            ToolCallFormat::DeepSeekV32 => Some("<｜DSML｜function_calls>"),
            ToolCallFormat::MiniMax => Some("<minimax:tool_call>"),
            ToolCallFormat::MiniMaxM3 => Some(M3_OPEN),
            ToolCallFormat::Gemma4 => Some("<|tool_call>"),
            // A harmony call opens with a channel header, which also
            // opens ordinary messages -- there is no marker that means
            // "tool call" on its own.
            ToolCallFormat::GptOss => None,
            // Nor does muse-glimmer have one: `<atem:function_calls>`
            // is a tool call only inside a tool channel, and the same
            // text in an answer is prose. A checkpoint anchor set on it
            // would fire on the model quoting its own instructions.
            ToolCallFormat::MuseGlimmer => None,
        }
    }

    /// Whether this format's arguments stream as prefix-stable
    /// fragments, or arrive whole when the call completes.
    ///
    /// MiniMax-M3 is deliberately not in this list: its arguments are a
    /// recursive element tree whose meaning depends on siblings that
    /// have not arrived yet (a repeated tag turns a value into an
    /// array), so a fragment emitted early would have to be taken back.
    /// Its calls are emitted whole, one per `</invoke>`.
    pub fn streams_arguments(self) -> bool {
        matches!(
            self,
            ToolCallFormat::Qwen3Coder
                | ToolCallFormat::DeepSeekV32
                | ToolCallFormat::MiniMax
                | ToolCallFormat::Glm47
                | ToolCallFormat::MuseGlimmer
        )
    }

    /// Whether this format's parser has to be told where the turn
    /// started.
    ///
    /// Only the channel grammar does: muse-glimmer's templated prompt
    /// ends *inside* an assistant channel header, so a parser reading
    /// the raw turn bytes must start header-open. Downstream of the
    /// reasoning parser it must not -- that parser delivers tool slices
    /// with their full headers already attached.
    pub fn accepts_header_open(self) -> bool {
        matches!(self, ToolCallFormat::MuseGlimmer)
    }

    /// This format's framing, the one description both the reader and
    /// the writer of a call work from. See [`Markers`].
    pub(crate) fn markers(self) -> Markers {
        match self {
            ToolCallFormat::Qwen25 => Markers::block("<tool_call>", "</tool_call>"),
            ToolCallFormat::Llama3 => Markers::block("<|python_tag|>", ""),
            ToolCallFormat::Mistral => Markers::block("[TOOL_CALLS]", ""),
            ToolCallFormat::Gemma4 => Markers::block(gemma::BLOCK_OPEN, gemma::BLOCK_CLOSE),
            ToolCallFormat::GptOss => Markers::block("<|channel|>", "<|call|>"),
            // The element grammar is scanned by `parse_m3` rather than
            // by the shared invoke machinery, so these tags are read
            // straight off the `M3_*` constants there. They are spelled
            // out here anyway because `tool_grammar` WRITES a forced
            // call from them, and the same constants on both sides is
            // what keeps the writer and the reader one description.
            ToolCallFormat::MiniMaxM3 => Markers {
                open: M3_OPEN,
                close: M3_CLOSE,
                invoke: Some(TagGrammar {
                    open: M3_INVOKE_OPEN,
                    name: NameStyle::Attribute,
                    close: M3_INVOKE_CLOSE,
                }),
                param: Some(TagGrammar {
                    open: M3_TAG,
                    name: NameStyle::Element,
                    close: M3_CLOSE_TAG,
                }),
                // `m3_typed_leaf` hands a declared string over verbatim
                // and trims only before it parses a number, which is
                // what `convert_declared` does for `TrimStyle::None`.
                trim_newlines: TrimStyle::None,
                // An element the schema never mentioned reaches
                // `parse_loose` there.
                undeclared: Undeclared::Loose,
            },
            ToolCallFormat::MuseGlimmer => Markers {
                open: "<atem:function_calls>",
                close: "</atem:function_calls>",
                invoke: Some(TagGrammar {
                    open: "<atem:invoke",
                    name: NameStyle::Attribute,
                    close: "</atem:invoke>",
                }),
                param: Some(TagGrammar {
                    open: "<atem:parameter",
                    name: NameStyle::Attribute,
                    close: "</atem:parameter>",
                }),
                // "spaces for string values are not stripped" is the
                // chat template's own words.
                trim_newlines: TrimStyle::None,
                undeclared: Undeclared::String,
            },
            ToolCallFormat::Qwen3Coder => Markers {
                open: "<tool_call>",
                close: "</tool_call>",
                invoke: Some(TagGrammar {
                    open: "<function=",
                    name: NameStyle::Bare,
                    close: "</function>",
                }),
                param: Some(TagGrammar {
                    open: "<parameter=",
                    name: NameStyle::Bare,
                    close: "</parameter>",
                }),
                trim_newlines: TrimStyle::One,
                undeclared: Undeclared::Loose,
            },
            ToolCallFormat::MiniMax => Markers {
                open: "<minimax:tool_call>",
                close: "</minimax:tool_call>",
                invoke: Some(TagGrammar {
                    open: "<invoke",
                    name: NameStyle::Attribute,
                    close: "</invoke>",
                }),
                param: Some(TagGrammar {
                    open: "<parameter",
                    name: NameStyle::Attribute,
                    close: "</parameter>",
                }),
                trim_newlines: TrimStyle::All,
                undeclared: Undeclared::Loose,
            },
            ToolCallFormat::DeepSeekV32 => Markers {
                open: "<｜DSML｜function_calls>",
                close: "</｜DSML｜function_calls>",
                invoke: Some(TagGrammar {
                    open: "<｜DSML｜invoke",
                    name: NameStyle::Attribute,
                    close: "</｜DSML｜invoke>",
                }),
                param: Some(TagGrammar {
                    open: "<｜DSML｜parameter",
                    name: NameStyle::Attribute,
                    close: "</｜DSML｜parameter>",
                }),
                trim_newlines: TrimStyle::None,
                undeclared: Undeclared::Loose,
            },
            ToolCallFormat::Glm47 => Markers {
                open: "<tool_call>",
                close: "</tool_call>",
                invoke: None,
                param: Some(TagGrammar {
                    open: "<arg_key>",
                    name: NameStyle::Paired {
                        key_close: "</arg_key>",
                        value_open: "<arg_value>",
                    },
                    close: "</arg_value>",
                }),
                trim_newlines: TrimStyle::All,
                undeclared: Undeclared::Loose,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NameStyle {
    /// The name is everything between the opener and `>`.
    Bare,
    /// The name is the `name="…"` attribute.
    Attribute,
    /// The name is an element of its own, and the value opens in a
    /// second element after it: GLM writes
    /// `<arg_key>k</arg_key><arg_value>v</arg_value>`.
    ///
    /// A variant rather than a `param.open == "<arg_key>"` test inside
    /// [`ToolCallParser::read_param_header`], because that test was a
    /// second place the format was written down: everything else about
    /// GLM lives in [`Markers`], and a reader that special-cased a
    /// marker string is one a writer cannot derive itself from.
    Paired {
        /// Closes the key element.
        key_close: &'static str,
        /// Opens the value element. [`TagGrammar::close`] closes it.
        value_open: &'static str,
    },
    /// The name IS the element: MiniMax-M3 writes
    /// `]<]minimax[>[<city>Rome]<]minimax[>[</city>`, so
    /// [`TagGrammar::open`] and [`TagGrammar::close`] are the two
    /// PREFIXES the name is written after, each finished with a `>`.
    ///
    /// The only style whose closing tag depends on the argument, which
    /// is why `TagGrammar::close` cannot be used as a literal for it.
    Element,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TagGrammar {
    pub(crate) open: &'static str,
    pub(crate) name: NameStyle,
    pub(crate) close: &'static str,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TrimStyle {
    /// Values are handed over exactly as written.
    None,
    /// Strip at most one leading and one trailing newline -- the ones
    /// the template's own layout inserts.
    One,
    /// Strip all surrounding whitespace.
    All,
}

/// What a parameter the request's schema never mentioned is worth.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Undeclared {
    /// Guess from the text: a number if it reads as one, and so on.
    Loose,
    /// A string, always. muse-glimmer's template says so, and guessing
    /// against it turns an undeclared `007` into `7`.
    String,
}

/// One format's framing, as data.
///
/// This is the single description of a wire format in this server: the
/// parser reads by it, and [`crate::tool_grammar`] writes the GBNF root
/// rule that FORCES a call from the same values. Two hand-maintained
/// tables -- one to read a framing and one to write it -- is this
/// repo's dominant bug shape, so there is one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Markers {
    pub(crate) open: &'static str,
    pub(crate) close: &'static str,
    pub(crate) invoke: Option<TagGrammar>,
    pub(crate) param: Option<TagGrammar>,
    pub(crate) trim_newlines: TrimStyle,
    pub(crate) undeclared: Undeclared,
}

impl Markers {
    const fn block(open: &'static str, close: &'static str) -> Self {
        Markers {
            open,
            close,
            invoke: None,
            param: None,
            trim_newlines: TrimStyle::None,
            undeclared: Undeclared::Loose,
        }
    }
}

/// Which ATEM channel the muse-glimmer parser is inside.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channel {
    /// Outside a channel, or inside one addressed to the user: text.
    Text,
    /// Inside a `to=self` channel: reasoning the tool parser has no
    /// business emitting. Only reached when no reasoning parser is
    /// stacked above, which is exactly when it must be swallowed.
    Skip,
    /// Inside a channel addressed to a tool: ATEM markup, executed.
    Tool,
}

/// Parses one response's tool calls.
#[derive(Debug)]
pub struct ToolCallParser {
    format: ToolCallFormat,
    markers: Markers,
    tools: BTreeMap<String, ToolSchema>,
    buffer: String,
    /// Calls emitted so far, which is also the next call's index.
    emitted: usize,
    /// Set while a streamed call is open.
    open: Option<OpenCall>,
    /// MiniMax-M3 only: whether the buffer's head is a wrapper that has
    /// opened and not closed. While it is, the buffer *keeps* the
    /// opener, so a stream that ends mid-block is still recognizable as
    /// a block and its complete invokes can be salvaged.
    m3_in_block: bool,
    /// muse-glimmer only: whether this parser reads the raw turn, whose
    /// prompt ended inside a channel header.
    header_open: bool,
    /// muse-glimmer only: the channel the stream is in.
    channel: Channel,
    /// muse-glimmer only: whether the `<|start|>` at the head of the
    /// buffer is the seed rather than model output, and so must be
    /// dropped rather than delivered if it turns out not to be a
    /// header.
    synthetic_open: bool,
    /// muse-glimmer only: whether a channel boundary cut an invoke in
    /// half. What follows is that broken channel's markup residue, not
    /// content, until the first character that cannot be residue.
    truncated_channel: bool,
    /// Llama 3 only: whether the stream's head is still undecided
    /// between prose and a bare-JSON call (`{"name": …}` with no
    /// `<|python_tag|>`). Cleared by the first non-blank character that
    /// is not `{`; while set, nothing is released as text.
    bare_head: bool,
    /// Llama 3 only: the head was held as a possible bare-JSON call, so
    /// if it parses to no call it is released as the text it was.
    held_bare: bool,
}

#[derive(Debug)]
struct OpenCall {
    index: usize,
    name: String,
    /// The arguments JSON built so far, without its closing brace.
    ledger: String,
    /// Whether any parameter has been written into the ledger yet.
    started: bool,
}

impl ToolCallParser {
    pub fn new(format: ToolCallFormat, tools: Vec<ToolSchema>) -> Self {
        ToolCallParser::with_header_open(format, tools, false)
    }

    /// A parser told where the turn started.
    ///
    /// Pass `header_open` for a muse-glimmer checkpoint whose raw turn
    /// bytes reach this parser directly -- the template's prompt ends
    /// with `<|start|>assistant`, so the model's first token continues
    /// a header nobody sent us. A synthetic `<|start|>` is seeded so
    /// those bytes go through the ordinary full-header machinery
    /// instead of a second guessing path; if the candidate turns out
    /// not to be a header, the seed is dropped rather than emitted.
    ///
    /// Every other format ignores the flag
    /// ([`ToolCallFormat::accepts_header_open`]).
    pub fn with_header_open(
        format: ToolCallFormat,
        tools: Vec<ToolSchema>,
        header_open: bool,
    ) -> Self {
        let header_open = header_open && format.accepts_header_open();
        ToolCallParser {
            format,
            markers: format.markers(),
            tools: tools
                .into_iter()
                .map(|tool| (tool.name.clone(), tool))
                .collect(),
            buffer: if header_open {
                ATEM_START.to_string()
            } else {
                String::new()
            },
            emitted: 0,
            open: None,
            m3_in_block: false,
            header_open,
            channel: Channel::Text,
            synthetic_open: header_open,
            truncated_channel: false,
            bare_head: format == ToolCallFormat::Llama3,
            held_bare: false,
        }
    }

    pub fn format(&self) -> ToolCallFormat {
        self.format
    }

    /// Whether `text` could contain a call in *any* known format.
    ///
    /// Deliberately format-agnostic and cheap: a response with none of
    /// these markers needs no parsing at all, and that is the common
    /// case.
    pub fn text_may_contain_call(text: &str) -> bool {
        TOOL_MARKERS.iter().any(|marker| text.contains(marker))
    }

    /// Whether `text` contains a call in *this* format.
    pub fn has_tool_call(&self, text: &str) -> bool {
        match self.format {
            ToolCallFormat::GptOss => {
                text.contains("to=functions.") && text.contains("<|message|>")
            }
            ToolCallFormat::Llama3 => {
                text.contains("<|python_tag|>") || text.trim_start().starts_with('{')
            }
            ToolCallFormat::Qwen3Coder => {
                text.contains("<function=") || text.contains("<tool_call>")
            }
            _ => text.contains(self.markers.open),
        }
    }

    /// Parse a complete response into its text and its calls.
    pub fn parse_complete(&self, text: &str) -> (String, Vec<ToolCall>) {
        if self.tools.is_empty() {
            return (text.to_string(), Vec::new());
        }
        // muse-glimmer takes no such shortcut: its channel layer
        // classifies text as well as calls, and a `to=self` body with
        // no ATEM block in it would otherwise be handed to the client
        // as the answer.
        if self.format == ToolCallFormat::MuseGlimmer {
            return self.replay_streaming(text);
        }
        if !self.has_tool_call(text) {
            return (text.to_string(), Vec::new());
        }
        match self.format {
            ToolCallFormat::MiniMaxM3 => self.parse_m3(text),
            ToolCallFormat::MuseGlimmer => unreachable!("handled above"),
            ToolCallFormat::Qwen25 => self.parse_json_blocks(text, "<tool_call>", "</tool_call>"),
            ToolCallFormat::Llama3 => self.parse_bare_json(text, "<|python_tag|>"),
            ToolCallFormat::Mistral => self.parse_mistral(text),
            ToolCallFormat::GptOss => self.parse_harmony(text),
            ToolCallFormat::Gemma4 => self.parse_gemma(text),
            ToolCallFormat::Glm47 => self.parse_glm(text),
            ToolCallFormat::Qwen3Coder | ToolCallFormat::MiniMax | ToolCallFormat::DeepSeekV32 => {
                self.parse_invoke(text)
            }
        }
    }

    /// Feed one increment and get the events it completed, in wire
    /// order.
    pub fn push(&mut self, chunk: &str) -> Vec<ToolCallEvent> {
        self.buffer.push_str(chunk);
        let mut events = Vec::new();
        if self.tools.is_empty() {
            let text = std::mem::take(&mut self.buffer);
            if !text.is_empty() {
                events.push(ToolCallEvent::Text(text));
            }
            return events;
        }
        match self.format {
            // Two grammars whose structure is not a marker pair: they
            // drive their own scan.
            ToolCallFormat::MiniMaxM3 => self.drain_m3(&mut events),
            ToolCallFormat::MuseGlimmer => self.drain_muse(&mut events),
            _ => loop {
                let progressed = if self.open.is_some() {
                    self.step_open_call(&mut events)
                } else {
                    self.step_idle(&mut events)
                };
                if !progressed {
                    break;
                }
            },
        }
        events
    }

    /// Release whatever is still buffered when the stream ends.
    ///
    /// A call left half-written by a truncated generation is closed
    /// with what it has, so the fragments a client already received
    /// still concatenate to valid JSON. Silently dropping it would
    /// leave the client parsing an unterminated object forever.
    pub fn finish(&mut self) -> Vec<ToolCallEvent> {
        let mut events = Vec::new();
        if self.format == ToolCallFormat::MuseGlimmer {
            self.finish_muse(&mut events);
            return events;
        }
        if let Some(open) = self.open.take() {
            let arguments = close_ledger(&open);
            events.push(ToolCallEvent::CallArguments {
                index: open.index,
                fragment: if open.started {
                    "}".into()
                } else {
                    "{}".into()
                },
            });
            events.push(ToolCallEvent::CallEnd {
                index: open.index,
                arguments,
            });
            self.emitted = open.index + 1;
        }
        let rest = std::mem::take(&mut self.buffer);
        if rest.is_empty() {
            return events;
        }
        if !self.has_tool_call(&rest) {
            self.emit_text(rest, &mut events);
            return events;
        }
        // Llama 3 and Mistral have no closing marker, so their payload
        // is only parseable once the stream ends -- this is the only
        // point at which their calls can be emitted at all. Dropping
        // the buffer here because it "looks like a call" would lose
        // every call those two families make.
        let (text, calls) = self.parse_complete(&rest);
        // A held bare-JSON head that names no offered tool was an
        // answer that happened to be JSON: it goes out as written.
        if calls.is_empty() && self.held_bare {
            self.emit_text(rest, &mut events);
            return events;
        }
        if !text.trim().is_empty() {
            events.push(ToolCallEvent::Text(text));
        }
        for call in calls {
            let index = self.emitted;
            self.emitted += 1;
            events.push(ToolCallEvent::CallStart {
                index,
                name: call.name,
            });
            events.push(ToolCallEvent::CallArguments {
                index,
                fragment: call.arguments.clone(),
            });
            events.push(ToolCallEvent::CallEnd {
                index,
                arguments: call.arguments,
            });
        }
        events
    }

    /// Looking for the start of a call. Returns whether it made
    /// progress.
    fn step_idle(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        // A model does not always wrap its call in the block marker its
        // template shows -- Qwen3-Coder in particular often emits a
        // bare `<function=…>`. The invoke tag opens a call just as
        // definitively, so whichever comes first counts.
        let opener = match self.markers.invoke {
            Some(invoke) => [self.markers.open, invoke.open]
                .iter()
                .filter_map(|marker| self.buffer.find(marker))
                .min(),
            None => self.buffer.find(self.markers.open),
        };
        match opener {
            Some(index) => {
                if index > 0 {
                    let text: String = self.buffer.drain(..index).collect();
                    self.emit_text(text, events);
                }
                if self.format.streams_arguments() {
                    self.start_streamed_block(events)
                } else {
                    self.take_complete_block(events)
                }
            }
            None => {
                // Llama 3 may write its call as bare JSON with no
                // marker. It has no closing marker either, so a turn
                // that opens with `{` is held whole and decided at
                // `finish` -- releasing it piecemeal would deliver the
                // call to the client as text before it could be read.
                if self.bare_head {
                    match self.buffer.trim_start().chars().next() {
                        None => return false,
                        Some('{') => {
                            self.held_bare = true;
                            return false;
                        }
                        Some(_) => self.bare_head = false,
                    }
                }
                // Withhold any trailing run that could still grow into
                // a marker -- opening OR closing. The closing ones
                // matter because a streamed invoke grammar leaves its
                // block close behind after the call is consumed, and at
                // one character per chunk it would otherwise dribble
                // out as `<`, `/`, `t`, ... before anything could
                // recognize it as structure.
                let markers: Vec<String> = TOOL_MARKERS
                    .iter()
                    .map(|marker| marker.to_string())
                    .chain(self.closing_markers().into_iter().map(str::to_string))
                    .collect();
                let hold = stop_prefix_holdback(&self.buffer, &markers);
                let split = floor_char_boundary(&self.buffer, self.buffer.len() - hold);
                if split == 0 {
                    return false;
                }
                let text: String = self.buffer.drain(..split).collect();
                self.emit_text(text, events);
                false
            }
        }
    }

    /// Release text, minus any structure that outlived its call.
    ///
    /// A streamed call is consumed marker by marker as it is
    /// recognized, and the *block* close can be left over -- an invoke
    /// grammar closes `</function>` and then `</tool_call>`, and only
    /// the first belongs to the call. Emitting the remainder as content
    /// would print raw markup into an answer, which is exactly what a
    /// client renders verbatim.
    fn emit_text(&self, text: String, events: &mut Vec<ToolCallEvent>) {
        let mut text = text;
        for marker in self.closing_markers() {
            if text.contains(marker) {
                text = text.replace(marker, "");
            }
        }
        if !text.is_empty() {
            events.push(ToolCallEvent::Text(text));
        }
    }

    /// This format's closing markers: structure that can outlive the
    /// call it closed.
    fn closing_markers(&self) -> Vec<&'static str> {
        [
            Some(self.markers.close),
            self.markers.invoke.map(|tag| tag.close),
            self.markers.param.map(|tag| tag.close),
        ]
        .into_iter()
        .flatten()
        .filter(|marker| !marker.is_empty())
        .collect()
    }

    /// A non-streaming format: wait for the whole block, then emit the
    /// calls it holds.
    fn take_complete_block(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        let close = self.markers.close;
        // Formats with no closing marker (Llama 3, Mistral) only become
        // parseable at the end of the stream.
        if close.is_empty() {
            return false;
        }
        let Some(end) = self.buffer.find(close) else {
            return false;
        };
        let block: String = self.buffer.drain(..end + close.len()).collect();
        let (_, calls) = self.parse_complete(&block);
        for call in calls {
            let index = self.emitted;
            self.emitted += 1;
            events.push(ToolCallEvent::CallStart {
                index,
                name: call.name,
            });
            events.push(ToolCallEvent::CallArguments {
                index,
                fragment: call.arguments.clone(),
            });
            events.push(ToolCallEvent::CallEnd {
                index,
                arguments: call.arguments,
            });
        }
        true
    }

    /// A streaming format: open a call as soon as its name is known.
    fn start_streamed_block(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        let Some(invoke) = self.markers.invoke else {
            return self.start_glm_call(events);
        };
        let Some(start) = self.buffer.find(invoke.open) else {
            // The block opened but the first invoke has not arrived. If
            // the block closed with nothing in it, drop it.
            if let Some(end) = self.buffer.find(self.markers.close) {
                self.buffer.drain(..end + self.markers.close.len());
                return true;
            }
            return false;
        };
        let attrs_start = start + invoke.open.len();
        let Some(gt) = self.buffer[attrs_start..]
            .find('>')
            .map(|i| i + attrs_start)
        else {
            return false;
        };
        let attrs = &self.buffer[attrs_start..gt];
        let Some(name) = read_name(attrs, invoke.name) else {
            // A malformed opener: drop it rather than stalling.
            self.buffer.drain(..gt + 1);
            return true;
        };
        let name = self.canonical_name(name);
        let index = self.emitted;
        self.open = Some(OpenCall {
            index,
            name: name.clone(),
            ledger: String::new(),
            started: false,
        });
        self.buffer.drain(..gt + 1);
        events.push(ToolCallEvent::CallStart { index, name });
        true
    }

    /// GLM's grammar names the function in bare text after the block
    /// opener rather than in a tag.
    fn start_glm_call(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        let open = self.markers.open;
        let Some(start) = self.buffer.find(open) else {
            return false;
        };
        let after = start + open.len();
        let rest = &self.buffer[after..];
        // The name ends at the first newline, the first argument, or
        // the end of the block -- whichever comes first.
        let end = ["\n", "<arg_key>", "</tool_call>"]
            .iter()
            .filter_map(|marker| rest.find(marker))
            .min();
        let Some(end) = end else {
            return false;
        };
        let name = rest[..end].trim().to_string();
        if name.is_empty() {
            return false;
        }
        let index = self.emitted;
        self.open = Some(OpenCall {
            index,
            name: name.clone(),
            ledger: String::new(),
            started: false,
        });
        self.buffer.drain(..after + end);
        events.push(ToolCallEvent::CallStart { index, name });
        true
    }

    /// Inside a call: consume one parameter, or close it.
    fn step_open_call(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        let param = self
            .markers
            .param
            .expect("a streaming format has parameters");
        let invoke_close = self.markers.invoke.map(|tag| tag.close);
        let block_close = self.markers.close;

        let param_at = self.buffer.find(param.open);
        let end_at = invoke_close
            .and_then(|close| self.buffer.find(close))
            .or_else(|| self.buffer.find(block_close));

        // A close before the next parameter ends the call.
        if let Some(end) = end_at {
            if param_at.is_none_or(|p| end < p) {
                let close_len = invoke_close
                    .filter(|close| self.buffer[end..].starts_with(close))
                    .map(str::len)
                    .unwrap_or(block_close.len());
                self.buffer.drain(..end + close_len);
                let open = self.open.take().expect("a call is open");
                let arguments = close_ledger(&open);
                events.push(ToolCallEvent::CallArguments {
                    index: open.index,
                    fragment: if open.started {
                        "}".into()
                    } else {
                        "{}".into()
                    },
                });
                events.push(ToolCallEvent::CallEnd {
                    index: open.index,
                    arguments,
                });
                self.emitted = open.index + 1;
                return true;
            }
        }

        let Some(start) = param_at else {
            return false;
        };
        let (key, is_string_attr, value_start) = match self.read_param_header(start, &param) {
            Some(header) => header,
            None => return false,
        };
        // The value runs to its closing tag; without one it is still
        // arriving.
        let Some(value_end) = self.buffer[value_start..]
            .find(param.close)
            .map(|i| i + value_start)
        else {
            return false;
        };
        let raw = self.buffer[value_start..value_end].to_string();
        self.buffer.drain(..value_end + param.close.len());

        let name = self.open.as_ref().expect("a call is open").name.clone();
        let value = self.convert_value(&name, &key, &raw, is_string_attr);
        let fragment = {
            let open = self.open.as_mut().expect("a call is open");
            let lead = if open.started { "," } else { "{" };
            let fragment = format!(
                "{lead}{}:{}",
                Value::String(key),
                serde_json::to_string(&value).unwrap_or_else(|_| "null".into())
            );
            open.ledger.push_str(&fragment);
            open.started = true;
            fragment
        };
        let index = self.open.as_ref().expect("a call is open").index;
        events.push(ToolCallEvent::CallArguments { index, fragment });
        true
    }

    /// Read a parameter opener at `start`: its key, whether the wire
    /// marked it as a string, and where its value begins.
    fn read_param_header(&self, start: usize, param: &TagGrammar) -> Option<(String, bool, usize)> {
        match param.name {
            // GLM writes `<arg_key>k</arg_key><arg_value>v`.
            NameStyle::Paired {
                key_close,
                value_open,
            } => {
                let key_start = start + param.open.len();
                let key_end = self.buffer[key_start..].find(key_close)? + key_start;
                let key = self.buffer[key_start..key_end].trim().to_string();
                let value_start =
                    self.buffer[key_end..].find(value_open)? + key_end + value_open.len();
                Some((key, false, value_start))
            }
            NameStyle::Bare => {
                let attrs_start = start + param.open.len();
                let gt = self.buffer[attrs_start..].find('>')? + attrs_start;
                let key = self.buffer[attrs_start..gt].trim().to_string();
                Some((key, false, gt + 1))
            }
            // MiniMax-M3's elements are read by `drain_m3` /
            // `m3_read_element`, which depth-match a value's closing tag
            // against the name that opened it. This machinery cannot:
            // it looks for one FIXED closing tag, and M3's depends on
            // the argument. Reaching here would mean M3 had been routed
            // through the shared invoke path, which `push` and
            // `parse_complete` both send to `drain_m3` instead.
            NameStyle::Element => None,
            NameStyle::Attribute => {
                let attrs_start = start + param.open.len();
                let gt = self.buffer[attrs_start..].find('>')? + attrs_start;
                let attrs = &self.buffer[attrs_start..gt];
                let key = read_name(attrs, NameStyle::Attribute)?;
                // DeepSeek marks a value that must stay a string, which
                // beats whatever the schema says.
                let is_string = read_attribute(attrs, "string").as_deref() == Some("true");
                Some((key, is_string, gt + 1))
            }
        }
    }

    /// Type one XML-ish parameter value.
    ///
    /// Precedence: an explicit wire marker, then the tool's declared
    /// schema, then a best-effort JSON parse. A value the schema calls
    /// a string is never parsed, which is what keeps `"018956"` and
    /// `"1.0"` intact.
    fn convert_value(&self, tool: &str, key: &str, raw: &str, wire_says_string: bool) -> Value {
        let trimmed = match self.markers.trim_newlines {
            TrimStyle::None => raw,
            TrimStyle::One => raw
                .strip_prefix('\n')
                .unwrap_or(raw)
                .strip_suffix('\n')
                .unwrap_or_else(|| raw.strip_prefix('\n').unwrap_or(raw)),
            TrimStyle::All => raw.trim(),
        };
        if wire_says_string {
            return Value::String(trimmed.to_string());
        }
        let declared = self.tools.get(tool).and_then(|t| t.parameter_type(key));
        if declared.is_none() && self.markers.undeclared == Undeclared::String {
            return Value::String(trimmed.to_string());
        }
        convert_declared(trimmed, declared.as_deref())
    }

    // ---- one-shot parsers, per family ----

    fn parse_json_blocks(&self, text: &str, open: &str, close: &str) -> (String, Vec<ToolCall>) {
        let mut normal = String::new();
        let mut calls = Vec::new();
        let mut cursor = 0usize;
        while let Some(start) = text[cursor..].find(open).map(|i| i + cursor) {
            normal.push_str(&text[cursor..start]);
            let body_start = start + open.len();
            let Some(end) = text[body_start..].find(close).map(|i| i + body_start) else {
                normal.push_str(&text[start..]);
                cursor = text.len();
                break;
            };
            if let Some(call) = self.call_from_json(text[body_start..end].trim(), calls.len()) {
                calls.push(call);
            }
            cursor = end + close.len();
        }
        normal.push_str(&text[cursor..]);
        (normal, calls)
    }

    /// Llama 3: one or more bare JSON objects after a marker, or a
    /// response that simply starts with one.
    fn parse_bare_json(&self, text: &str, marker: &str) -> (String, Vec<ToolCall>) {
        let (normal, payload) = match text.find(marker) {
            Some(index) => (text[..index].to_string(), &text[index + marker.len()..]),
            None => (String::new(), text),
        };
        let mut calls = Vec::new();
        let mut rest = payload.trim_start();
        while rest.starts_with('{') {
            let Some(end) = json_object_end(rest) else {
                break;
            };
            if let Some(call) = self.call_from_json(&rest[..end], calls.len()) {
                calls.push(call);
            }
            rest = rest[end..]
                .trim_start()
                .trim_start_matches([';', ','])
                .trim_start();
        }
        (normal, calls)
    }

    fn parse_mistral(&self, text: &str) -> (String, Vec<ToolCall>) {
        let marker = "[TOOL_CALLS]";
        let Some(index) = text.find(marker) else {
            return (text.to_string(), Vec::new());
        };
        let normal = text[..index].to_string();
        let rest = text[index + marker.len()..].trim_start();
        let Some(end) = json_array_end(rest) else {
            return (text.to_string(), Vec::new());
        };
        let Ok(Value::Array(items)) = serde_json::from_str::<Value>(&rest[..end]) else {
            return (text.to_string(), Vec::new());
        };
        let mut calls = Vec::new();
        for item in items {
            if let Some(call) = self.call_from_value(&item, calls.len()) {
                calls.push(call);
            }
        }
        (normal, calls)
    }

    /// gpt-oss: a commentary channel addressed to a function, whose
    /// message body is the arguments JSON.
    fn parse_harmony(&self, text: &str) -> (String, Vec<ToolCall>) {
        let mut normal = String::new();
        let mut calls = Vec::new();
        let mut cursor = 0usize;
        while let Some(channel) = text[cursor..]
            .find(harmony::CHANNEL_OPEN)
            .map(|i| i + cursor)
        {
            let header_start = channel + harmony::CHANNEL_OPEN.len();
            let Some(message) = text[header_start..]
                .find(harmony::MESSAGE_OPEN)
                .map(|i| i + header_start)
            else {
                break;
            };
            let header = &text[header_start..message];
            let body_start = message + harmony::MESSAGE_OPEN.len();
            let (end, matched) = harmony::BODY_ENDS
                .iter()
                .filter_map(|marker| {
                    text[body_start..]
                        .find(marker)
                        .map(|i| (i + body_start, *marker))
                })
                .min_by_key(|(index, _)| *index)
                .map(|(index, marker)| (index, Some(marker)))
                .unwrap_or((text.len(), None));

            if let Some(name) = header.split_whitespace().find_map(|token| {
                token
                    .strip_prefix(harmony::RECIPIENT_KEY)?
                    .strip_prefix(harmony::FUNCTION_NAMESPACE)
            }) {
                normal.push_str(&text[cursor..channel]);
                let arguments = normalize_arguments(&text[body_start..end]);
                if self.known(name) {
                    calls.push(ToolCall {
                        index: calls.len(),
                        name: name.to_string(),
                        arguments,
                    });
                }
            } else {
                normal.push_str(&text[cursor..end]);
            }
            cursor = match matched {
                Some(marker) => end + marker.len(),
                None => text.len(),
            };
        }
        normal.push_str(&text[cursor..]);
        (normal, calls)
    }

    /// Gemma 4: `call:name{key: value, …}` with its own quoting.
    fn parse_gemma(&self, text: &str) -> (String, Vec<ToolCall>) {
        let mut normal = String::new();
        let mut calls = Vec::new();
        let mut cursor = 0usize;
        while let Some(start) = text[cursor..].find(gemma::BLOCK_OPEN).map(|i| i + cursor) {
            normal.push_str(&text[cursor..start]);
            let body_start = start + gemma::BLOCK_OPEN.len();
            let Some(end) = text[body_start..]
                .find(gemma::BLOCK_CLOSE)
                .map(|i| i + body_start)
            else {
                normal.push_str(&text[start..]);
                cursor = text.len();
                break;
            };
            let body = text[body_start..end].trim();
            if let Some(rest) = body.strip_prefix(gemma::CALL_KEY) {
                if let Some(brace) = rest.find(gemma::ARGS_OPEN) {
                    let name = rest[..brace].trim();
                    let args = rest[brace + gemma::ARGS_OPEN.len()..]
                        .trim_end()
                        .trim_end_matches(gemma::ARGS_CLOSE);
                    if self.known(name) {
                        calls.push(ToolCall {
                            index: calls.len(),
                            name: name.to_string(),
                            arguments: gemma_arguments(args),
                        });
                    }
                }
            }
            cursor = end + gemma::BLOCK_CLOSE.len();
        }
        normal.push_str(&text[cursor..]);
        (normal, calls)
    }

    fn parse_glm(&self, text: &str) -> (String, Vec<ToolCall>) {
        self.replay_streaming(text)
    }

    fn parse_invoke(&self, text: &str) -> (String, Vec<ToolCall>) {
        self.replay_streaming(text)
    }

    /// Run the streaming machinery on a fresh parser.
    ///
    /// The invoke families have exactly one implementation of their
    /// grammar, so a streamed response and a buffered one cannot
    /// disagree about what the model said.
    fn replay_streaming(&self, text: &str) -> (String, Vec<ToolCall>) {
        let mut parser = ToolCallParser::with_header_open(
            self.format,
            self.tools.values().cloned().collect(),
            self.header_open,
        );
        let mut events = parser.push(text);
        events.extend(parser.finish());

        let mut normal = String::new();
        let mut calls = Vec::new();
        let mut names: BTreeMap<usize, String> = BTreeMap::new();
        for event in events {
            match event {
                ToolCallEvent::Text(text) => normal.push_str(&text),
                ToolCallEvent::CallStart { index, name } => {
                    names.insert(index, name);
                }
                ToolCallEvent::CallArguments { .. } => {}
                ToolCallEvent::CallEnd { index, arguments } => {
                    if let Some(name) = names.remove(&index) {
                        if self.known(&name) {
                            calls.push(ToolCall {
                                index: calls.len(),
                                name,
                                arguments,
                            });
                        }
                    }
                }
            }
        }
        (normal, calls)
    }

    fn call_from_json(&self, text: &str, index: usize) -> Option<ToolCall> {
        let value: Value = serde_json::from_str(text).ok()?;
        self.call_from_value(&value, index)
    }

    fn call_from_value(&self, value: &Value, index: usize) -> Option<ToolCall> {
        let name = value.get("name")?.as_str()?.to_string();
        if !self.known(&name) {
            return None;
        }
        // Families disagree on the key; they never both appear.
        let arguments = value
            .get("arguments")
            .or_else(|| value.get("parameters"))
            .cloned()
            .unwrap_or(Value::Object(Map::new()));
        Some(ToolCall {
            index,
            name,
            arguments: serde_json::to_string(&arguments).unwrap_or_else(|_| "{}".into()),
        })
    }

    /// Whether the request actually offered this tool.
    ///
    /// A name the request never offered is dropped: forwarding it makes
    /// the client execute something it did not advertise. A namespaced
    /// name (`skills:read`) is forwarded anyway -- those are routed by
    /// the client, which is where the namespace is resolved.
    fn known(&self, name: &str) -> bool {
        self.tools.contains_key(name) || name.contains(':')
    }

    /// Normalize an invoke's function name before it is reported.
    ///
    /// muse-glimmer's template renders a bare-name tool's recipient
    /// namespace as `name.*`, so the model writes
    /// `get_weather.get_weather`. Collapsing the doubled form to the
    /// registered head is what makes such a call executable -- but
    /// never when the doubled form is itself a registered tool, and
    /// never for a genuinely namespaced `weather.get`.
    fn canonical_name(&self, name: String) -> String {
        if self.format != ToolCallFormat::MuseGlimmer {
            return name;
        }
        if let Some((head, tail)) = name.split_once('.') {
            if head == tail && self.tools.contains_key(head) && !self.tools.contains_key(&name) {
                return head.to_string();
            }
        }
        name
    }

    // ---- MiniMax-M3: a namespaced recursive element grammar ----

    /// Scan the buffer for M3 wrappers, emitting one call per complete
    /// `</invoke>`.
    ///
    /// While a wrapper is open the buffer *keeps* its opener, so a
    /// stream that stops mid-block is still recognizable as a block:
    /// [`ToolCallParser::finish`] re-parses it and salvages every
    /// invoke that did complete.
    fn drain_m3(&mut self, events: &mut Vec<ToolCallEvent>) {
        while !self.buffer.is_empty() {
            if !self.m3_in_block {
                match self.buffer.find(M3_OPEN) {
                    None => {
                        // The hold-back has to take the LONGEST suffix
                        // that is a proper prefix of the opener, not
                        // the shortest: `]` recurs inside
                        // `]<]minimax[>[`, so a buffer ending `]<]`
                        // matches at one character too, and holding
                        // only that leaks `]<` as content -- after
                        // which the marker can never be recognized.
                        let hold = longest_partial_prefix(&self.buffer, M3_OPEN);
                        let split = floor_char_boundary(&self.buffer, self.buffer.len() - hold);
                        if split == 0 {
                            return;
                        }
                        let text: String = self.buffer.drain(..split).collect();
                        events.push(ToolCallEvent::Text(text));
                        return;
                    }
                    Some(0) => self.m3_in_block = true,
                    Some(at) => {
                        let text: String = self.buffer.drain(..at).collect();
                        events.push(ToolCallEvent::Text(text));
                    }
                }
                continue;
            }
            // Inside a wrapper. Markers are interpreted structurally --
            // between elements only -- so a value quoting the wire
            // syntax can neither spawn a phantom call nor end the block.
            let step = {
                let body = &self.buffer[M3_OPEN.len()..];
                match m3_next_action(body) {
                    None => None,
                    Some(M3Action::Close { at }) => Some(M3Step::Close { at }),
                    Some(M3Action::Invoke { name, body_at }) => {
                        let (items, end, closer) = m3_scan_invoke_interior(body, body_at);
                        // Anything but a complete `</invoke>` means the
                        // call is still arriving (or was truncated, in
                        // which case `finish` salvages it).
                        (closer == Some(M3Closer::Invoke)).then_some(M3Step::Call {
                            name,
                            items,
                            end,
                        })
                    }
                }
            };
            match step {
                None => return,
                Some(M3Step::Close { at }) => {
                    self.buffer = self.buffer[M3_OPEN.len() + at + M3_CLOSE.len()..].to_string();
                    self.m3_in_block = false;
                }
                Some(M3Step::Call { name, items, end }) => {
                    let arguments = self.m3_arguments(&name, &items);
                    let rest =
                        self.buffer[M3_OPEN.len() + end + M3_INVOKE_CLOSE.len()..].to_string();
                    self.buffer = format!("{M3_OPEN}{rest}");
                    let index = self.emitted;
                    self.emitted += 1;
                    events.push(ToolCallEvent::CallStart { index, name });
                    events.push(ToolCallEvent::CallArguments {
                        index,
                        fragment: arguments.clone(),
                    });
                    events.push(ToolCallEvent::CallEnd { index, arguments });
                }
            }
        }
    }

    /// One-shot: every wrapper in the text, with the text between and
    /// after them kept as content.
    fn parse_m3(&self, text: &str) -> (String, Vec<ToolCall>) {
        let mut normal = String::new();
        let mut calls = Vec::new();
        let mut pos = 0usize;
        loop {
            let Some(at) = text[pos..].find(M3_OPEN).map(|i| i + pos) else {
                normal.push_str(&text[pos..]);
                break;
            };
            normal.push_str(&text[pos..at]);
            pos = self.parse_m3_block(text, at + M3_OPEN.len(), &mut calls);
        }
        (normal, calls)
    }

    /// Parse one wrapper interior, returning where it ended.
    ///
    /// A wrapper that never closes -- the shape a generation truncated
    /// by `max_tokens` leaves -- still yields the invokes that did
    /// complete, and a truncated trailing element still salvages its
    /// complete siblings.
    fn parse_m3_block(&self, text: &str, from: usize, calls: &mut Vec<ToolCall>) -> usize {
        let mut pos = from;
        while pos < text.len() {
            let Some(at) = text[pos..].find(M3_TAG).map(|i| i + pos) else {
                return text.len();
            };
            if text[at..].starts_with(M3_CLOSE) {
                return at + M3_CLOSE.len();
            }
            let Some((name, body_at)) = m3_invoke_open_at(text, at) else {
                pos = at + M3_NS.len() + 1; // a stray marker at block level
                continue;
            };
            let (items, end, closer) = m3_scan_invoke_interior(text, body_at);
            if self.known(&name) {
                let arguments = self.m3_arguments(&name, &items);
                calls.push(ToolCall {
                    index: calls.len(),
                    name,
                    arguments,
                });
            }
            match closer {
                Some(M3Closer::Invoke) => pos = end + M3_INVOKE_CLOSE.len(),
                Some(M3Closer::Wrapper) => return end + M3_CLOSE.len(),
                None => return text.len(),
            }
        }
        text.len()
    }

    /// Build one invoke's arguments from its scanned elements.
    ///
    /// The tool's schema is threaded down the whole tree, not just
    /// applied at the top: a nested leaf declared `string` stays a
    /// string, which is what keeps a postcode like `018956` from being
    /// read as the number 18956 three levels down.
    fn m3_arguments(&self, name: &str, items: &[(String, String)]) -> String {
        let params = self
            .tools
            .get(name)
            .and_then(|tool| tool.parameters.as_ref());
        // The properties map when there is one, the schema itself
        // otherwise -- some requests send the properties bare.
        let props = params.map(|schema| schema.get("properties").unwrap_or(schema));
        let value = m3_args_from_items(items, props);
        serde_json::to_string(&value).unwrap_or_else(|_| "{}".into())
    }

    // ---- muse-glimmer: an ATEM channel around an invoke block ----

    /// Run the channel layer, which decides what the invoke machinery
    /// is even allowed to look at.
    fn drain_muse(&mut self, events: &mut Vec<ToolCallEvent>) {
        while !self.buffer.is_empty() {
            let progressed = match self.channel {
                Channel::Text => self.muse_text(events),
                Channel::Skip => self.muse_skip(),
                Channel::Tool => self.muse_tool(events),
            };
            if !progressed {
                break;
            }
        }
    }

    /// Outside a tool channel: stream text and watch for a header.
    fn muse_text(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        if self.truncated_channel {
            // A broken channel's trailing markup is dropped BY SHAPE --
            // closing tags and whitespace -- and the mark clears at the
            // first character that cannot be residue. Waiting for a
            // later boundary instead would swallow the turn's real
            // reply, which on this wire arrives immediately.
            let (consumed, decided) = atem_residue_end(&self.buffer);
            if consumed > 0 {
                self.buffer.drain(..consumed);
            }
            if !decided {
                return false;
            }
            self.truncated_channel = false;
            return true;
        }
        let Some(boundary) = atem_boundary(&self.buffer) else {
            let hold = atem_hold_len(&self.buffer);
            let split = floor_char_boundary(&self.buffer, self.buffer.len() - hold);
            if split == 0 {
                return false;
            }
            let text: String = self.buffer.drain(..split).collect();
            events.push(ToolCallEvent::Text(text));
            return false;
        };
        let at = boundary.at();
        if at > 0 {
            let text: String = self.buffer.drain(..at).collect();
            events.push(ToolCallEvent::Text(text));
            return true;
        }
        match boundary {
            // A terminator with no channel to close is protocol debris.
            AtemBoundary::Closer { token, .. } => {
                self.buffer.drain(..token.len());
                true
            }
            AtemBoundary::Switch(switch) => {
                self.buffer.drain(..switch.end);
                self.enter_recipient(&switch.recipient);
                true
            }
            AtemBoundary::Start { .. } => self.muse_header(events),
        }
    }

    /// Read the channel header the buffer opens with.
    fn muse_header(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        let synthetic = self.synthetic_open;
        let message = self.buffer.find(ATEM_MESSAGE);
        // A control token inside the candidate settles it at once:
        // headers never contain one, so this `<|start|>` is text.
        if atem_marker_inside(
            &self.buffer,
            ATEM_START.len(),
            message.unwrap_or(self.buffer.len()),
        ) {
            self.release_start(synthetic, events);
            return true;
        }
        match message {
            // Too far away to belong to this marker: a stray literal
            // `<|start|>`, junk, then the next segment's real header.
            Some(message) if message - ATEM_START.len() > ATEM_HEADER_SPAN => {
                self.release_start(synthetic, events);
                true
            }
            Some(message) => {
                let recipient = atem_recipient(&self.buffer[ATEM_START.len()..message])
                    .unwrap_or_else(|| "user".into());
                self.synthetic_open = false;
                self.buffer.drain(..message + ATEM_MESSAGE.len());
                self.enter_recipient(&recipient);
                true
            }
            None => {
                // Held while it could still become a header, with slack
                // for a `<|message|>` mid-arrival; past that bound it
                // cannot open one any more and is released as text, so
                // a degenerate wire neither stalls nor eats the turn.
                if self.buffer.len() > ATEM_START.len() + ATEM_HEADER_SPAN + ATEM_MESSAGE.len() {
                    self.release_start(synthetic, events);
                    return true;
                }
                false
            }
        }
    }

    /// Give up on a `<|start|>` that opens no header. The synthetic
    /// seed is dropped rather than delivered: the model never wrote it.
    fn release_start(&mut self, synthetic: bool, events: &mut Vec<ToolCallEvent>) {
        if !synthetic {
            events.push(ToolCallEvent::Text(ATEM_START.to_string()));
        }
        self.synthetic_open = false;
        self.buffer.drain(..ATEM_START.len());
    }

    /// Inside a `to=self` channel: swallow the body.
    fn muse_skip(&mut self) -> bool {
        match atem_boundary(&self.buffer) {
            None => {
                let hold = atem_hold_len(&self.buffer);
                let split = floor_char_boundary(&self.buffer, self.buffer.len() - hold);
                self.buffer.drain(..split);
                false
            }
            Some(boundary) => {
                // Drop the body; the boundary itself is handled by text
                // mode, exactly as if the channel had been one.
                self.buffer.drain(..boundary.at());
                self.channel = Channel::Text;
                true
            }
        }
    }

    /// Inside a tool channel: the body up to the next channel boundary
    /// belongs to the invoke machinery.
    fn muse_tool(&mut self, events: &mut Vec<ToolCallEvent>) -> bool {
        let Some(boundary) = atem_boundary(&self.buffer) else {
            let hold = atem_hold_len(&self.buffer);
            let split = floor_char_boundary(&self.buffer, self.buffer.len() - hold);
            if split == 0 {
                return false;
            }
            return self.run_atem_machinery(split, events);
        };
        let at = boundary.at();
        if self.run_atem_machinery(at, events) {
            // The machinery stops after each call, so a second block
            // before the same boundary must get its turn rather than
            // being written off as debris.
            return true;
        }
        // Inert: nothing before the boundary the machinery can consume.
        // Close a mid-flight invoke here rather than letting the next
        // channel's call merge into it, then leave the channel; the
        // residue ahead of the boundary is markup debris and goes with
        // it.
        self.finalize_truncated_invoke(events);
        match boundary {
            AtemBoundary::Closer { token, .. } => {
                self.buffer.drain(..at + token.len());
                self.channel = Channel::Text;
            }
            AtemBoundary::Switch(switch) => {
                self.buffer.drain(..switch.end);
                self.enter_recipient(&switch.recipient);
            }
            AtemBoundary::Start { .. } => {
                self.buffer.drain(..at);
                self.channel = Channel::Text;
            }
        }
        true
    }

    /// Feed the first `prefix_len` bytes of the buffer through the
    /// ordinary invoke/parameter machinery, keeping the rest buffered
    /// behind it. Returns whether it consumed anything.
    ///
    /// Text the machinery releases inside a tool channel is dropped:
    /// only tool-recipient markup is executed, and the template puts
    /// nothing else in such a channel.
    fn run_atem_machinery(&mut self, prefix_len: usize, events: &mut Vec<ToolCallEvent>) -> bool {
        let remainder = self.buffer.split_off(prefix_len);
        let before = self.buffer.len();
        let mut scratch = Vec::new();
        loop {
            let progressed = if self.open.is_some() {
                self.step_open_call(&mut scratch)
            } else {
                self.step_idle(&mut scratch)
            };
            if !progressed {
                break;
            }
        }
        let consumed = self.buffer.len() != before;
        self.buffer.push_str(&remainder);
        events.extend(
            scratch
                .into_iter()
                .filter(|event| !matches!(event, ToolCallEvent::Text(_))),
        );
        consumed
    }

    /// Close a call the channel (or the stream) cut off mid-arguments.
    ///
    /// Its `CallStart` has already gone out, so the client is holding a
    /// half-written arguments object. Closing it here is what keeps the
    /// fragments it received concatenating to valid JSON; dropping the
    /// call instead would leave that client parsing an unterminated
    /// object forever, and reusing its ordinal would merge the next
    /// channel's call into this one.
    fn finalize_truncated_invoke(&mut self, events: &mut Vec<ToolCallEvent>) {
        let Some(open) = self.open.take() else {
            return;
        };
        let arguments = close_ledger(&open);
        events.push(ToolCallEvent::CallArguments {
            index: open.index,
            fragment: if open.started {
                "}".into()
            } else {
                "{}".into()
            },
        });
        events.push(ToolCallEvent::CallEnd {
            index: open.index,
            arguments,
        });
        self.emitted = open.index + 1;
        self.truncated_channel = true;
    }

    /// End of stream for the channel format.
    fn finish_muse(&mut self, events: &mut Vec<ToolCallEvent>) {
        self.finalize_truncated_invoke(events);
        let mut residual = std::mem::take(&mut self.buffer);
        let channel = self.channel;
        let synthetic = self.synthetic_open;
        let truncated = self.truncated_channel;
        self.buffer = if self.header_open {
            ATEM_START.to_string()
        } else {
            String::new()
        };
        self.channel = Channel::Text;
        self.synthetic_open = self.header_open;
        self.truncated_channel = false;
        if channel != Channel::Text || truncated {
            // Whatever is left is the inside of a channel that never
            // closed: markup, or a body meant for somebody who is not
            // the client.
            return;
        }
        if synthetic {
            if let Some(rest) = residual.strip_prefix(ATEM_START) {
                residual = rest.to_string();
            }
        }
        // At end of stream a `<|start|>` that never received its
        // `<|message|>` is not a header: deliver the text, drop only
        // the markers.
        residual = residual.replace(ATEM_START, "");
        for token in ATEM_CLOSING_TOKENS {
            residual = residual.replace(token, "");
        }
        if self.emitted > 0 && residual.trim().is_empty() {
            return;
        }
        if !residual.is_empty() {
            events.push(ToolCallEvent::Text(residual));
        }
    }

    /// Enter the channel a header just named.
    fn enter_recipient(&mut self, recipient: &str) {
        self.channel = match recipient {
            "self" => Channel::Skip,
            "user" => Channel::Text,
            _ => Channel::Tool,
        };
    }
}

/// One thing the M3 scanner can do next.
#[derive(Debug)]
enum M3Action {
    Close { at: usize },
    Invoke { name: String, body_at: usize },
}

/// One thing the M3 streaming loop can do next.
#[derive(Debug)]
enum M3Step {
    Close {
        at: usize,
    },
    Call {
        name: String,
        items: Vec<(String, String)>,
        end: usize,
    },
}

/// What ended an invoke's interior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum M3Closer {
    Invoke,
    Wrapper,
}

/// One read of the element scanner.
#[derive(Debug)]
enum M3Element {
    /// A complete `NS<name>…NS</name>`.
    Element {
        name: String,
        inner: String,
        next: usize,
    },
    /// Model noise -- a dangling closer, or a tag whose name cannot be
    /// one. Scanning continues at `next`: aborting would drop every
    /// well-formed sibling after it.
    Skipped { next: usize },
    /// The element never terminated. Everything before it still counts,
    /// which is what makes a call cut off by `max_tokens` salvageable.
    Truncated,
}

/// The longest suffix of `text` that is a proper prefix of `marker`.
///
/// Deliberately longest, not shortest: a marker whose own characters
/// recur inside it (`]` in `]<]minimax[>[`) matches at several lengths,
/// and holding back the shortest one emits the rest as content.
fn longest_partial_prefix(text: &str, marker: &str) -> usize {
    let max = marker.len().saturating_sub(1).min(text.len());
    (1..=max)
        .rev()
        .find(|&k| {
            let at = text.len() - k;
            text.is_char_boundary(at) && marker.starts_with(&text[at..])
        })
        .unwrap_or(0)
}

/// Read one element at `pos`, which must be the start of an M3 tag.
fn m3_read_element(text: &str, pos: usize) -> M3Element {
    if text[pos..].starts_with(M3_CLOSE_TAG) {
        return M3Element::Skipped {
            next: pos + M3_CLOSE_TAG.len(),
        };
    }
    let tag_start = pos + M3_TAG.len();
    let Some(gt) = text[tag_start..].find('>').map(|i| i + tag_start) else {
        return M3Element::Truncated;
    };
    let name = &text[tag_start..gt];
    if name.is_empty() || name.contains(M3_TAG_BAD) {
        return M3Element::Skipped { next: gt + 1 };
    }
    let open_tag = format!("{M3_NS}<{name}>");
    let close_tag = format!("{M3_NS}</{name}>");
    // Same-name nesting is depth-matched, so an `<items>` inside an
    // `<items>` closes the right one.
    let mut depth = 1usize;
    let mut search = gt + 1;
    let mut inner_end = gt + 1;
    while depth > 0 {
        let Some(close_at) = text[search..].find(&close_tag).map(|i| i + search) else {
            return M3Element::Truncated;
        };
        match text[search..].find(&open_tag).map(|i| i + search) {
            Some(open_at) if open_at < close_at => {
                depth += 1;
                search = open_at + open_tag.len();
            }
            _ => {
                depth -= 1;
                inner_end = close_at;
                search = close_at + close_tag.len();
            }
        }
    }
    M3Element::Element {
        name: name.to_string(),
        inner: text[gt + 1..inner_end].to_string(),
        next: search,
    }
}

/// Scan a value for nested elements, returning them and the text
/// between them.
///
/// Lenient on purpose. The strict alternative -- voiding the whole
/// parameter over one stray character -- loses a call the model got
/// right in every respect but one. The cost is stated: a leaf value
/// that quotes a well-formed element pair parses as structure.
fn m3_scan_elements(text: &str) -> (Vec<(String, String)>, String) {
    let mut items = Vec::new();
    let mut stray = String::new();
    let mut pos = 0usize;
    while pos < text.len() {
        let Some(at) = text[pos..].find(M3_TAG).map(|i| i + pos) else {
            stray.push_str(&text[pos..]);
            break;
        };
        stray.push_str(&text[pos..at]);
        match m3_read_element(text, at) {
            M3Element::Element { name, inner, next } => {
                items.push((name, inner));
                pos = next;
            }
            M3Element::Skipped { next } => pos = next,
            M3Element::Truncated => break,
        }
    }
    (items, stray)
}

/// Collect an invoke's parameter elements, from just past its opener to
/// its closer.
///
/// Close markers count only *between* elements, which is what stops a
/// value quoting `]<]minimax[>[</tool_call>` from ending the block.
fn m3_scan_invoke_interior(
    text: &str,
    from: usize,
) -> (Vec<(String, String)>, usize, Option<M3Closer>) {
    let mut items = Vec::new();
    let mut pos = from;
    while pos < text.len() {
        let Some(at) = text[pos..].find(M3_TAG).map(|i| i + pos) else {
            return (items, text.len(), None);
        };
        if text[at..].starts_with(M3_INVOKE_CLOSE) {
            return (items, at, Some(M3Closer::Invoke));
        }
        if text[at..].starts_with(M3_CLOSE) {
            return (items, at, Some(M3Closer::Wrapper));
        }
        match m3_read_element(text, at) {
            M3Element::Element { name, inner, next } => {
                items.push((name, inner));
                pos = next;
            }
            M3Element::Skipped { next } => pos = next,
            // Truncated: salvage the complete siblings.
            M3Element::Truncated => return (items, text.len(), None),
        }
    }
    (items, text.len(), None)
}

/// The next structural thing in a wrapper interior: its closer, or an
/// invoke opener. A marker that is neither is model noise and is
/// stepped over.
fn m3_next_action(body: &str) -> Option<M3Action> {
    let mut scan = 0usize;
    while let Some(at) = body[scan..].find(M3_TAG).map(|i| i + scan) {
        if body[at..].starts_with(M3_CLOSE) {
            return Some(M3Action::Close { at });
        }
        if let Some((name, body_at)) = m3_invoke_open_at(body, at) {
            return Some(M3Action::Invoke { name, body_at });
        }
        scan = at + M3_NS.len() + 1;
    }
    None
}

/// Read `]<]minimax[>[<invoke name="…">` at `at`, giving the name and
/// where the interior begins.
///
/// The template renders double quotes; models emit single ones too, so
/// both are accepted.
fn m3_invoke_open_at(text: &str, at: usize) -> Option<(String, usize)> {
    let rest = text.get(at..)?.strip_prefix(M3_INVOKE_OPEN)?;
    let attrs = rest.trim_start_matches([' ', '\t', '\r', '\n']);
    if attrs.len() == rest.len() {
        return None; // the opener needs whitespace before its attributes
    }
    let value = attrs.strip_prefix("name=")?;
    let quote = value.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let body = &value[quote.len_utf8()..];
    let end = body.find(quote)?;
    let name = &body[..end];
    if name.is_empty() {
        return None;
    }
    let tail = body[end + quote.len_utf8()..].trim_start_matches([' ', '\t', '\r', '\n']);
    let tail = tail.strip_prefix('>')?;
    Some((name.to_string(), text.len() - tail.len()))
}

/// Build an invoke's arguments object from its top-level elements.
fn m3_args_from_items(items: &[(String, String)], props: Option<&Value>) -> Value {
    let mut out = Map::new();
    for (key, raw) in items {
        let prop = props.and_then(|p| p.get(key));
        let (nested, stray) = m3_scan_elements(raw);
        let value = if nested.is_empty() {
            m3_typed_leaf(raw, prop)
        } else {
            m3_structure(&nested, prop, &stray)
        };
        m3_insert(&mut out, key, value);
    }
    Value::Object(out)
}

/// Insert a key, turning a repeat into an array rather than losing the
/// first occurrence.
fn m3_insert(map: &mut Map<String, Value>, key: &str, value: Value) {
    match map.remove(key) {
        Some(Value::Array(mut items)) => {
            items.push(value);
            map.insert(key.to_string(), Value::Array(items));
        }
        Some(previous) => {
            map.insert(key.to_string(), Value::Array(vec![previous, value]));
        }
        None => {
            map.insert(key.to_string(), value);
        }
    }
}

/// A value that has children: an object, or an array when the template
/// used its array convention.
///
/// Only `<item>` children render as a bare array -- that is the
/// convention the template writes. Repeated siblings under any other
/// tag stay an object with an array-valued key, because collapsing them
/// would drop the parent key the tool is expecting.
fn m3_structure(items: &[(String, String)], schema: Option<&Value>, stray: &str) -> Value {
    let props = schema.and_then(|s| s.get("properties"));
    let item_schema = schema.and_then(|s| s.get("items"));
    let sub_schema = |key: &str| -> Option<&Value> {
        props.and_then(|p| p.get(key)).or_else(|| {
            schema
                .and_then(|s| s.get("additionalProperties"))
                .filter(|value| value.is_object())
        })
    };
    if !items.is_empty() && items.iter().all(|(name, _)| name == "item") {
        return Value::Array(
            items
                .iter()
                .map(|(_, raw)| m3_nested_value(raw, item_schema))
                .collect(),
        );
    }
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for (name, _) in items {
        *counts.entry(name.as_str()).or_default() += 1;
    }
    let mut out = Map::new();
    for (key, raw) in items {
        let mut sub = sub_schema(key);
        // A repeated key's elements are the array's members, so its
        // array schema unwraps to the item schema. A singleton key
        // keeps the array schema -- its `<item>` children unwrap
        // inside it.
        if counts[key.as_str()] > 1 && declared_type(sub) == Some("array".to_string()) {
            sub = sub.and_then(|s| s.get("items"));
        }
        m3_insert(&mut out, key, m3_nested_value(raw, sub));
    }
    if !stray.trim().is_empty() {
        out.insert("$text".to_string(), Value::String(stray.trim().to_string()));
    }
    Value::Object(out)
}

/// One element's value: a subtree if it has children, a typed leaf
/// otherwise.
fn m3_nested_value(raw: &str, schema: Option<&Value>) -> Value {
    let (items, stray) = m3_scan_elements(raw);
    if items.is_empty() {
        m3_typed_leaf(raw, schema)
    } else {
        m3_structure(&items, schema, &stray)
    }
}

/// The normalized JSON Schema `type` of a subschema, if it states one.
fn declared_type(schema: Option<&Value>) -> Option<String> {
    let schema = schema?;
    if !schema.is_object() {
        return None;
    }
    Some(
        schema
            .get("type")?
            .as_str()?
            .trim()
            .to_ascii_lowercase()
            .to_string(),
    )
}

/// Type one leaf value from the schema at its own nesting level.
///
/// A string leaf round-trips **verbatim**: the template renders leaf
/// values exactly as written, and trimming here eats the trailing
/// newline of every multi-line file argument. Declared non-string types
/// tolerate the whitespace around them; an undeclared leaf falls back
/// to a loose parse but keeps its verbatim text whenever that parse
/// yields a string anyway.
fn m3_typed_leaf(raw: &str, prop: Option<&Value>) -> Value {
    let Some(prop) = prop else {
        if raw.trim().is_empty() {
            // An empty element is the empty STRING; a loose parse would
            // make it an empty object.
            return Value::String(String::new());
        }
        return match parse_loose(raw.trim()) {
            Value::String(_) => Value::String(raw.to_string()),
            value => value,
        };
    };
    if prop.is_object() && prop.get("type").is_none() {
        if let Some(Value::Array(subs)) = prop.get("anyOf").or_else(|| prop.get("oneOf")) {
            if !subs.is_empty() {
                return m3_union_leaf(raw, subs);
            }
        }
    }
    let declared = declared_type(Some(prop)).unwrap_or_else(|| "string".to_string());
    match declared.as_str() {
        "string" | "str" | "enum" => Value::String(raw.to_string()),
        // An integer literal stays an integer and `5.0` stays 5.0.
        "number" | "float" | "double" => {
            let text = raw.trim();
            if let Ok(value) = text.parse::<i64>() {
                Value::from(value)
            } else if let Ok(value) = text.parse::<f64>() {
                Value::from(value)
            } else {
                Value::String(raw.to_string())
            }
        }
        // An empty container-typed element is that empty container.
        "object" if raw.trim().is_empty() => Value::Object(Map::new()),
        "array" if raw.trim().is_empty() => Value::Array(Vec::new()),
        other => convert_declared(raw.trim(), Some(other)),
    }
}

/// An `anyOf` / `oneOf` leaf: try each member's strict coercion in the
/// order the schema declared them, and keep the verbatim text when only
/// the string member (or nothing) fits.
fn m3_union_leaf(raw: &str, subs: &[Value]) -> Value {
    let text = raw.trim();
    for sub in subs {
        match declared_type(Some(sub)).unwrap_or_default().as_str() {
            "integer" | "int" => {
                if let Ok(value) = text.parse::<i64>() {
                    return Value::from(value);
                }
            }
            "number" | "float" | "double" => {
                if let Ok(value) = text.parse::<i64>() {
                    return Value::from(value);
                }
                if let Ok(value) = text.parse::<f64>() {
                    return Value::from(value);
                }
            }
            "boolean" | "bool" => {
                if text.eq_ignore_ascii_case("true") || text.eq_ignore_ascii_case("false") {
                    return Value::Bool(text.eq_ignore_ascii_case("true"));
                }
            }
            "null" => {
                if text.eq_ignore_ascii_case("null") {
                    return Value::Null;
                }
            }
            "object" => {
                if let Ok(value @ Value::Object(_)) = serde_json::from_str::<Value>(text) {
                    return value;
                }
            }
            "array" => {
                if let Ok(value @ Value::Array(_)) = serde_json::from_str::<Value>(text) {
                    return value;
                }
            }
            _ => {}
        }
    }
    Value::String(raw.to_string())
}

/// The ATEM closing tags a broken-off invoke leaves behind.
const ATEM_CLOSING_MARKUP: [&str; 3] = [
    "</atem:parameter>",
    "</atem:invoke>",
    "</atem:function_calls>",
];

/// How much of `text` is the leading run of ATEM closing markup, and
/// whether the scan *decided*.
///
/// `false` means the buffer ends inside the run or on a partial tag, so
/// more input is needed. The cost of dropping by shape is stated: a
/// reply that literally begins with a complete closing tag loses that
/// tag. Partial-tag-shaped prose is held and released intact, so no
/// ordinary word is ever eaten -- only a verbatim full tag immediately
/// after a truncated invoke, which is indistinguishable from the
/// residue this exists to drop.
fn atem_residue_end(text: &str) -> (usize, bool) {
    let mut at = 0usize;
    while at < text.len() {
        let rest = &text[at..];
        let ch = rest.chars().next().expect("the slice is non-empty");
        if matches!(ch, ' ' | '\t' | '\r' | '\n') {
            at += ch.len_utf8();
            continue;
        }
        if let Some(tag) = ATEM_CLOSING_MARKUP
            .iter()
            .find(|tag| rest.starts_with(**tag))
        {
            at += tag.len();
            continue;
        }
        if ATEM_CLOSING_MARKUP.iter().any(|tag| tag.starts_with(rest)) {
            return (at, false);
        }
        return (at, true);
    }
    (at, false)
}

fn close_ledger(open: &OpenCall) -> String {
    if open.started {
        format!("{}}}", open.ledger)
    } else {
        "{}".to_string()
    }
}

/// Type one value against the JSON Schema type its parameter declared.
///
/// `None` -- undeclared -- is a best-effort parse that keeps the text
/// whenever it is not obviously something else.
fn convert_declared(text: &str, declared: Option<&str>) -> Value {
    // A declared NUMBER carries no meaning in the whitespace around it,
    // and `str::parse` refuses it: a DeepSeek template's own layout puts
    // a newline there (`TrimStyle::None` keeps it, because a declared
    // STRING's spaces are the model's), and an un-trimmed `"\n3\n"`
    // reached the tool as a string where the schema said integer. The
    // string arms are deliberately not trimmed.
    match declared {
        Some("string") | Some("str") | Some("enum") => Value::String(text.to_string()),
        Some("integer") | Some("int") => text
            .trim()
            .parse::<i64>()
            .map(Value::from)
            .unwrap_or_else(|_| Value::String(text.to_string())),
        Some("number") | Some("float") | Some("double") => text
            .trim()
            .parse::<f64>()
            .map(Value::from)
            .unwrap_or_else(|_| Value::String(text.to_string())),
        Some("boolean") | Some("bool") => Value::Bool(text.trim().eq_ignore_ascii_case("true")),
        Some("object") | Some("array") => {
            serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_string()))
        }
        _ => parse_loose(text),
    }
}

/// Read a `name` out of a tag's attribute text.
fn read_name(attrs: &str, style: NameStyle) -> Option<String> {
    match style {
        NameStyle::Bare => {
            let name = attrs.trim();
            (!name.is_empty()).then(|| name.to_string())
        }
        NameStyle::Attribute => read_attribute(attrs, "name"),
        // Only a parameter tag is ever paired, and a paired parameter's
        // key is read by `read_param_header` before this is reached.
        NameStyle::Paired { .. } => None,
        // An element-named tag has no attribute text at all: its name is
        // the tag. `m3_read_element` reads it, and nothing routes an M3
        // tag through this function.
        NameStyle::Element => None,
    }
}

/// Read `key="value"` (or `key='value'`) out of attribute text.
fn read_attribute(attrs: &str, key: &str) -> Option<String> {
    let pattern = format!("{key}=");
    let start = attrs.find(&pattern)? + pattern.len();
    let rest = &attrs[start..];
    let quote = rest.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let body = &rest[quote.len_utf8()..];
    let end = body.find(quote)?;
    Some(body[..end].to_string())
}

/// Parse a value with no declared type: JSON if it plainly is JSON, the
/// text otherwise.
fn parse_loose(text: &str) -> Value {
    if text.is_empty() {
        return Value::String(String::new());
    }
    match text {
        "true" | "True" => return Value::Bool(true),
        "false" | "False" => return Value::Bool(false),
        "null" | "None" => return Value::Null,
        _ => {}
    }
    match serde_json::from_str::<Value>(text) {
        // A bare word parses as nothing; a quoted one parses as the
        // string it already was.
        Ok(Value::String(_)) | Err(_) => Value::String(text.to_string()),
        Ok(value) => value,
    }
}

/// Re-serialize an arguments payload, or wrap it if it is not an
/// object.
fn normalize_arguments(raw: &str) -> String {
    let trimmed = raw.trim();
    match serde_json::from_str::<Value>(trimmed) {
        Ok(value @ Value::Object(_)) => {
            serde_json::to_string(&value).unwrap_or_else(|_| "{}".into())
        }
        _ => "{}".to_string(),
    }
}

/// Gemma's `k: v, k: v` argument list, with [`gemma::QUOTE`] for quotes.
fn gemma_arguments(text: &str) -> String {
    let mut map = Map::new();
    for pair in split_top_level(text, gemma::PAIR_SEPARATOR) {
        let Some((key, value)) = pair.split_once(gemma::KEY_SEPARATOR) else {
            continue;
        };
        let key = key.trim().trim_matches('"').to_string();
        let value = value.trim();
        let parsed = match value
            .strip_prefix(gemma::QUOTE)
            .and_then(|v| v.strip_suffix(gemma::QUOTE))
        {
            Some(inner) => Value::String(inner.to_string()),
            None => parse_loose(value),
        };
        map.insert(key, parsed);
    }
    serde_json::to_string(&Value::Object(map)).unwrap_or_else(|_| "{}".into())
}

/// Split on `delimiter`, ignoring delimiters inside brackets or quotes.
fn split_top_level(text: &str, delimiter: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut quoted = false;
    let mut current = String::new();
    let mut chars = text.char_indices().peekable();
    while let Some((index, ch)) = chars.next() {
        if text[index..].starts_with(gemma::QUOTE) {
            quoted = !quoted;
            current.push_str(gemma::QUOTE);
            for _ in 0..gemma::QUOTE.chars().count() - 1 {
                chars.next();
            }
            continue;
        }
        if !quoted && text[index..].starts_with(delimiter) && depth == 0 {
            parts.push(std::mem::take(&mut current));
            for _ in 0..delimiter.chars().count() - 1 {
                chars.next();
            }
            continue;
        }
        if !quoted {
            match ch {
                '{' | '[' => depth += 1,
                '}' | ']' => depth -= 1,
                _ => {}
            }
        }
        current.push(ch);
    }
    if !current.trim().is_empty() {
        parts.push(current);
    }
    parts
}

/// The byte just past the JSON object starting at `text[0]`.
fn json_object_end(text: &str) -> Option<usize> {
    balanced_end(text, '{', '}')
}

fn json_array_end(text: &str) -> Option<usize> {
    balanced_end(text, '[', ']')
}

/// Scan a balanced bracket span, respecting JSON strings and escapes.
fn balanced_end(text: &str, open: char, close: char) -> Option<usize> {
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (index, ch) in text.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            c if c == open => depth += 1,
            c if c == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(index + ch.len_utf8());
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tools() -> Vec<ToolSchema> {
        vec![
            ToolSchema::with_parameters(
                "get_weather",
                json!({"type": "object", "properties": {
                    "city": {"type": "string"},
                    "days": {"type": "integer"}
                }}),
            ),
            ToolSchema::with_parameters(
                "write_file",
                json!({"type": "object", "properties": {
                    "path": {"type": "string"},
                    "contents": {"type": "string"},
                    "overwrite": {"type": "boolean"}
                }}),
            ),
        ]
    }

    fn parser(format: ToolCallFormat) -> ToolCallParser {
        ToolCallParser::new(format, tools())
    }

    /// Chunk by characters, not bytes: a token boundary never splits a
    /// codepoint, and the DSML markers are multi-byte.
    fn stream(parser: &mut ToolCallParser, text: &str, width: usize) -> Vec<ToolCallEvent> {
        let chars: Vec<char> = text.chars().collect();
        let mut events = Vec::new();
        for chunk in chars.chunks(width) {
            let piece: String = chunk.iter().collect();
            events.extend(parser.push(&piece));
        }
        events.extend(parser.finish());
        events
    }

    fn calls_of(events: &[ToolCallEvent]) -> Vec<(usize, String, String)> {
        let mut names: BTreeMap<usize, String> = BTreeMap::new();
        let mut out = Vec::new();
        for event in events {
            match event {
                ToolCallEvent::CallStart { index, name } => {
                    names.insert(*index, name.clone());
                }
                ToolCallEvent::CallEnd { index, arguments } => {
                    out.push((*index, names[index].clone(), arguments.clone()));
                }
                _ => {}
            }
        }
        out
    }

    fn text_of(events: &[ToolCallEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                ToolCallEvent::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn a_hermes_call_is_recognized_with_its_arguments() {
        let parser = parser(ToolCallFormat::Qwen25);
        let (text, calls) = parser.parse_complete(
            "Let me check.<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}</tool_call>",
        );
        assert_eq!(text, "Let me check.");
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Rome"}"#);
    }

    #[test]
    fn several_hermes_calls_keep_their_order() {
        let parser = parser(ToolCallFormat::Qwen25);
        let (_, calls) = parser.parse_complete(
            "<tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}</tool_call>\n\
             <tool_call>{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}</tool_call>",
        );
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].index, 0);
        assert_eq!(calls[1].arguments, r#"{"city":"Oslo"}"#);
    }

    /// A name the request never offered must not reach the client: it
    /// would be asked to run something it did not advertise.
    #[test]
    fn a_tool_the_request_never_offered_is_dropped() {
        let parser = parser(ToolCallFormat::Qwen25);
        let (_, calls) = parser
            .parse_complete("<tool_call>{\"name\": \"rm_rf\", \"arguments\": {}}</tool_call>");
        assert!(calls.is_empty());
    }

    /// ... but a namespaced name is forwarded, because the client is
    /// what resolves the namespace.
    #[test]
    fn a_namespaced_tool_is_forwarded() {
        let parser = parser(ToolCallFormat::Qwen25);
        let (_, calls) = parser.parse_complete(
            "<tool_call>{\"name\": \"skills:read\", \"arguments\": {}}</tool_call>",
        );
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "skills:read");
    }

    #[test]
    fn llama_and_mistral_payloads_are_recognized() {
        let llama = parser(ToolCallFormat::Llama3);
        let (text, calls) = llama.parse_complete(
            "sure<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Rome\"}}",
        );
        assert_eq!(text, "sure");
        assert_eq!(calls[0].arguments, r#"{"city":"Rome"}"#);

        let mistral = parser(ToolCallFormat::Mistral);
        let (text, calls) = mistral.parse_complete(
            "ok [TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}]",
        );
        assert_eq!(text, "ok ");
        assert_eq!(calls[0].arguments, r#"{"city":"Oslo"}"#);
    }

    /// Llama 3.1/3.2 answer a custom tool with bare JSON, no
    /// `<|python_tag|>`. Streamed at any width, that must still come out
    /// as a call -- it used to be released as text, chunk by chunk.
    #[test]
    fn a_bare_llama_call_streams_as_a_call() {
        let payload = "{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Rome\"}}";
        for width in [1, 3, 7, 64] {
            let mut llama = parser(ToolCallFormat::Llama3);
            let events = stream(&mut llama, payload, width);
            assert_eq!(
                calls_of(&events),
                vec![(
                    0,
                    "get_weather".to_string(),
                    r#"{"city":"Rome"}"#.to_string()
                )],
                "width {width}"
            );
            assert!(
                text_of(&events).trim().is_empty(),
                "width {width}: {events:?}"
            );
        }
    }

    /// The hold only covers a head that could be a call: JSON naming no
    /// offered tool, and ordinary prose, come out as written.
    #[test]
    fn a_bare_llama_head_that_is_not_a_call_stays_text() {
        let json = "{\"answer\": 42}";
        let mut llama = parser(ToolCallFormat::Llama3);
        let events = stream(&mut llama, json, 2);
        assert!(calls_of(&events).is_empty());
        assert_eq!(text_of(&events), json);

        let prose = "Rome is sunny {mostly}.";
        let mut llama = parser(ToolCallFormat::Llama3);
        let events = stream(&mut llama, prose, 2);
        assert!(calls_of(&events).is_empty());
        assert_eq!(text_of(&events), prose);
    }

    #[test]
    fn a_harmony_commentary_channel_is_a_call() {
        let parser = parser(ToolCallFormat::GptOss);
        let (text, calls) = parser.parse_complete(
            "<|channel|>commentary to=functions.get_weather<|message|>{\"city\": \"Rome\"}<|call|>",
        );
        assert!(text.is_empty(), "{text:?}");
        assert_eq!(calls[0].name, "get_weather");
        assert_eq!(calls[0].arguments, r#"{"city":"Rome"}"#);
    }

    #[test]
    fn a_gemma_call_uses_its_own_quoting() {
        let parser = parser(ToolCallFormat::Gemma4);
        let (_, calls) = parser.parse_complete(
            "<|tool_call>call:get_weather{city: <|\"|>Rome<|\"|>, days: 3}<tool_call|>",
        );
        assert_eq!(calls.len(), 1);
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(args["city"], json!("Rome"));
        assert_eq!(args["days"], json!(3));
    }

    #[test]
    fn a_qwen_coder_call_streams_its_arguments() {
        let mut parser = parser(ToolCallFormat::Qwen3Coder);
        let wire = "<tool_call><function=write_file>\
                    <parameter=path>\n/tmp/x\n</parameter>\
                    <parameter=contents>\nhello\n</parameter>\
                    </function></tool_call>";
        let events = stream(&mut parser, wire, 7);
        let calls = calls_of(&events);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].1, "write_file");
        let args: Value = serde_json::from_str(&calls[0].2).unwrap();
        assert_eq!(args["path"], json!("/tmp/x"));
        assert_eq!(args["contents"], json!("hello"));
    }

    /// The streaming contract: the fragments concatenate to exactly the
    /// final arguments, so a client can parse what it accumulated.
    #[test]
    fn streamed_fragments_concatenate_to_the_final_arguments() {
        for format in [
            ToolCallFormat::Qwen3Coder,
            ToolCallFormat::MiniMax,
            ToolCallFormat::DeepSeekV32,
            ToolCallFormat::Glm47,
            ToolCallFormat::MiniMaxM3,
            ToolCallFormat::MuseGlimmer,
        ] {
            let wire = wire_for(format);
            for width in [1usize, 5, 17, 4096] {
                let mut parser = parser(format);
                let events = stream(&mut parser, &wire, width);
                let calls = calls_of(&events);
                assert_eq!(calls.len(), 1, "{format:?} width {width}: {events:?}");

                let joined: String = events
                    .iter()
                    .filter_map(|e| match e {
                        ToolCallEvent::CallArguments { fragment, .. } => Some(fragment.as_str()),
                        _ => None,
                    })
                    .collect();
                assert_eq!(joined, calls[0].2, "{format:?} width {width}");
                serde_json::from_str::<Value>(&joined)
                    .unwrap_or_else(|e| panic!("{format:?} width {width}: {e} in {joined}"));
            }
        }
    }

    /// And streaming agrees with a buffered parse of the same wire.
    #[test]
    fn streaming_agrees_with_one_shot_for_every_invoke_family() {
        for format in [
            ToolCallFormat::Qwen3Coder,
            ToolCallFormat::MiniMax,
            ToolCallFormat::DeepSeekV32,
            ToolCallFormat::Glm47,
            ToolCallFormat::MiniMaxM3,
            ToolCallFormat::MuseGlimmer,
        ] {
            let wire = wire_for(format);
            let (_, complete) = parser(format).parse_complete(&wire);
            let mut streamed = parser(format);
            let events = stream(&mut streamed, &wire, 3);
            let calls = calls_of(&events);
            assert_eq!(calls.len(), complete.len(), "{format:?}");
            assert_eq!(calls[0].1, complete[0].name, "{format:?}");
            assert_eq!(calls[0].2, complete[0].arguments, "{format:?}");
        }
    }

    fn wire_for(format: ToolCallFormat) -> String {
        match format {
            ToolCallFormat::Qwen3Coder => "<tool_call><function=get_weather>\
                 <parameter=city>\nRome\n</parameter>\
                 <parameter=days>\n3\n</parameter>\
                 </function></tool_call>"
                .to_string(),
            // M3 is a different protocol from MiniMax-M2, not renamed
            // tags: every structural tag is namespaced, and a parameter
            // is named by its own ELEMENT rather than by an attribute.
            ToolCallFormat::MiniMaxM3 => "]<]minimax[>[<tool_call>\
                 ]<]minimax[>[<invoke name=\"get_weather\">\
                 ]<]minimax[>[<city>Rome]<]minimax[>[</city>\
                 ]<]minimax[>[<days>3]<]minimax[>[</days>\
                 ]<]minimax[>[</invoke>]<]minimax[>[</tool_call>"
                .to_string(),
            // The ATEM block does not stand alone: muse-glimmer's
            // channel layer classifies the segment, so the block has to
            // arrive inside a `to=tool` message or the parser is right
            // to treat it as text.
            ToolCallFormat::MuseGlimmer => "assistant to=tool<|message|>\
                 <atem:function_calls>\
                 <atem:invoke name=\"get_weather\">\
                 <atem:parameter name=\"city\">Rome</atem:parameter>\
                 <atem:parameter name=\"days\">3</atem:parameter>\
                 </atem:invoke></atem:function_calls><|eot|>"
                .to_string(),
            ToolCallFormat::MiniMax => "<minimax:tool_call>\
                 <invoke name=\"get_weather\">\
                 <parameter name=\"city\">Rome</parameter>\
                 <parameter name=\"days\">3</parameter>\
                 </invoke></minimax:tool_call>"
                .to_string(),
            ToolCallFormat::DeepSeekV32 => "<｜DSML｜function_calls>\
                 <｜DSML｜invoke name=\"get_weather\">\
                 <｜DSML｜parameter name=\"city\" string=\"true\">Rome</｜DSML｜parameter>\
                 <｜DSML｜parameter name=\"days\">3</｜DSML｜parameter>\
                 </｜DSML｜invoke></｜DSML｜function_calls>"
                .to_string(),
            ToolCallFormat::Glm47 => "<tool_call>get_weather\n\
                 <arg_key>city</arg_key><arg_value>Rome</arg_value>\
                 <arg_key>days</arg_key><arg_value>3</arg_value>\
                 </tool_call>"
                .to_string(),
            _ => unreachable!("only the streaming families have invoke wire"),
        }
    }

    /// A declared string parameter is handed over verbatim, so a
    /// zero-padded id survives.
    #[test]
    fn a_declared_string_is_never_reparsed() {
        let tools = vec![ToolSchema::with_parameters(
            "lookup",
            json!({"type": "object", "properties": {"id": {"type": "string"}}}),
        )];
        let parser = ToolCallParser::new(ToolCallFormat::MiniMax, tools);
        let (_, calls) = parser.parse_complete(
            "<minimax:tool_call><invoke name=\"lookup\">\
             <parameter name=\"id\">018956</parameter></invoke></minimax:tool_call>",
        );
        assert_eq!(calls[0].arguments, r#"{"id":"018956"}"#);
    }

    /// ... and a declared number stays a number, including a float
    /// written with a trailing zero.
    #[test]
    fn a_declared_number_keeps_its_type() {
        let tools = vec![ToolSchema::with_parameters(
            "scale",
            json!({"type": "object", "properties": {
                "factor": {"type": "number"},
                "count": {"type": "integer"},
                "enabled": {"type": "boolean"}
            }}),
        )];
        let parser = ToolCallParser::new(ToolCallFormat::MiniMax, tools);
        let (_, calls) = parser.parse_complete(
            "<minimax:tool_call><invoke name=\"scale\">\
             <parameter name=\"factor\">5.0</parameter>\
             <parameter name=\"count\">7</parameter>\
             <parameter name=\"enabled\">true</parameter>\
             </invoke></minimax:tool_call>",
        );
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert!(args["factor"].is_f64(), "{args}");
        assert_eq!(args["count"], json!(7));
        assert_eq!(args["enabled"], json!(true));
    }

    /// DeepSeek marks a value as a string on the wire, and that beats
    /// any guess.
    #[test]
    fn the_wire_string_marker_wins_over_a_guess() {
        let tools = vec![ToolSchema::new("run")];
        let parser = ToolCallParser::new(ToolCallFormat::DeepSeekV32, tools);
        let (_, calls) = parser.parse_complete(
            "<｜DSML｜function_calls><｜DSML｜invoke name=\"run\">\
             <｜DSML｜parameter name=\"cmd\" string=\"true\">123</｜DSML｜parameter>\
             </｜DSML｜invoke></｜DSML｜function_calls>",
        );
        assert_eq!(calls[0].arguments, r#"{"cmd":"123"}"#);
    }

    /// A file's contents must survive verbatim -- the case a coding
    /// agent lives on.
    #[test]
    fn a_multiline_value_survives_verbatim() {
        let wire = "<minimax:tool_call><invoke name=\"write_file\">\
                    <parameter name=\"path\">/tmp/x.rs</parameter>\
                    <parameter name=\"contents\">fn main() {\n    println!(\"hi\");\n}</parameter>\
                    </invoke></minimax:tool_call>";
        let (_, calls) = parser(ToolCallFormat::MiniMax).parse_complete(wire);
        let args: Value = serde_json::from_str(&calls[0].arguments).unwrap();
        assert_eq!(
            args["contents"],
            json!("fn main() {\n    println!(\"hi\");\n}")
        );
    }

    #[test]
    fn a_call_with_no_arguments_still_reports_an_object() {
        let tools = vec![ToolSchema::new("ping")];
        let parser = ToolCallParser::new(ToolCallFormat::MiniMax, tools);
        let (_, calls) = parser.parse_complete(
            "<minimax:tool_call><invoke name=\"ping\"></invoke></minimax:tool_call>",
        );
        assert_eq!(calls[0].arguments, "{}");
    }

    /// Text either side of a call must arrive as text, in order.
    #[test]
    fn text_around_a_call_is_preserved_in_order() {
        let mut parser = parser(ToolCallFormat::Qwen25);
        let events = stream(
            &mut parser,
            "before <tool_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call> after",
            6,
        );
        let text = text_of(&events);
        assert!(text.starts_with("before "), "{text:?}");
        assert!(text.ends_with(" after"), "{text:?}");
        assert_eq!(calls_of(&events).len(), 1);
    }

    /// No prefix of a marker may reach the wire as text, or a client
    /// renders `<tool` and then has to take it back.
    #[test]
    fn a_partial_marker_is_never_streamed_as_text() {
        let mut parser = parser(ToolCallFormat::Qwen25);
        let events = parser.push("hello <tool");
        assert_eq!(text_of(&events), "hello ");
        let events =
            parser.push("_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call>");
        assert_eq!(text_of(&events), "");
        assert_eq!(calls_of(&events).len(), 1);
    }

    /// A streamed invoke grammar closes twice -- the call, then the
    /// block -- and only the first belongs to the call. The leftover
    /// must not be printed into the answer.
    #[test]
    fn a_block_close_left_over_from_a_call_is_not_content() {
        let wire = "<tool_call><function=get_weather>\
                    <parameter=city>\nRome\n</parameter>\
                    </function></tool_call>";
        for width in [1usize, 6, 4096] {
            let mut parser = parser(ToolCallFormat::Qwen3Coder);
            let events = stream(&mut parser, wire, width);
            assert_eq!(calls_of(&events).len(), 1, "width {width}");
            assert_eq!(
                text_of(&events),
                "",
                "width {width}: structure leaked as content"
            );
        }
    }

    /// A held partial that turns out to be ordinary text is released,
    /// not dropped.
    #[test]
    fn a_partial_that_is_not_a_marker_comes_back() {
        let mut parser = parser(ToolCallFormat::Qwen25);
        let events = stream(&mut parser, "compare a <tool b", 4);
        assert_eq!(text_of(&events), "compare a <tool b");
    }

    /// A generation truncated mid-call still leaves the client with
    /// parseable JSON.
    #[test]
    fn a_truncated_call_is_closed_with_what_it_has() {
        let mut parser = parser(ToolCallFormat::MiniMax);
        let events = stream(
            &mut parser,
            "<minimax:tool_call><invoke name=\"get_weather\">\
             <parameter name=\"city\">Rome</parameter>",
            9,
        );
        let calls = calls_of(&events);
        assert_eq!(calls.len(), 1);
        serde_json::from_str::<Value>(&calls[0].2).expect("valid JSON despite truncation");
    }

    /// Llama 3 and Mistral have no closing marker, so the end of the
    /// stream is the only place their calls can be recognized. A parser
    /// that dropped its buffer there because it "looked like a call"
    /// would lose every call those two families make.
    #[test]
    fn a_format_with_no_closing_marker_still_emits_at_the_end() {
        for (format, wire) in [
            (
                ToolCallFormat::Llama3,
                "<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {\"city\": \"Rome\"}}",
            ),
            (
                ToolCallFormat::Mistral,
                "[TOOL_CALLS] [{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Rome\"}}]",
            ),
        ] {
            let mut parser = parser(format);
            let events = stream(&mut parser, wire, 6);
            let calls = calls_of(&events);
            assert_eq!(calls.len(), 1, "{format:?}: {events:?}");
            assert_eq!(calls[0].1, "get_weather", "{format:?}");
            assert_eq!(calls[0].2, r#"{"city":"Rome"}"#, "{format:?}");
        }
    }

    /// Qwen3-Coder often emits a bare `<function=…>` without the
    /// `<tool_call>` wrapper its template shows. The invoke tag opens a
    /// call just as definitively.
    #[test]
    fn an_unwrapped_invoke_tag_still_opens_a_call() {
        let wire = "sure: <function=get_weather><parameter=city>\nRome\n</parameter></function>";
        let mut streamed = parser(ToolCallFormat::Qwen3Coder);
        // Push everything but do NOT finish: the call must be
        // recognized while the stream is running, not salvaged from the
        // buffer at the end.
        let mut events = Vec::new();
        for chunk in wire.chars().collect::<Vec<_>>().chunks(5) {
            events.extend(streamed.push(&chunk.iter().collect::<String>()));
        }
        assert!(
            events
                .iter()
                .any(|e| matches!(e, ToolCallEvent::CallStart { .. })),
            "the call must open mid-stream: {events:?}"
        );
        events.extend(streamed.finish());
        let calls = calls_of(&events);
        assert_eq!(calls.len(), 1, "{events:?}");
        assert_eq!(calls[0].2, r#"{"city":"Rome"}"#);
        assert_eq!(text_of(&events), "sure: ");

        let (_, complete) = parser(ToolCallFormat::Qwen3Coder).parse_complete(wire);
        assert_eq!(complete.len(), 1);
        assert_eq!(complete[0].arguments, r#"{"city":"Rome"}"#);
    }

    #[test]
    fn a_response_with_no_tools_offered_is_all_text() {
        let mut parser = ToolCallParser::new(ToolCallFormat::Qwen25, Vec::new());
        let wire = "<tool_call>{\"name\": \"get_weather\", \"arguments\": {}}</tool_call>";
        let events = stream(&mut parser, wire, 8);
        assert_eq!(text_of(&events), wire);
        assert!(calls_of(&events).is_empty());
        assert_eq!(parser.parse_complete(wire).1.len(), 0);
    }

    /// The inference order is load-bearing: the specific families must
    /// win over the general ones they resemble.
    #[test]
    fn format_inference_prefers_the_specific_family() {
        assert_eq!(
            ToolCallFormat::infer("Qwen3-Coder-30B"),
            ToolCallFormat::Qwen3Coder
        );
        assert_eq!(ToolCallFormat::infer("Qwen2.5-7B"), ToolCallFormat::Qwen25);
        assert_eq!(
            ToolCallFormat::infer("DeepSeek-V4-Flash"),
            ToolCallFormat::DeepSeekV32
        );
        assert_eq!(ToolCallFormat::infer("GLM-5.2"), ToolCallFormat::Glm47);
        assert_eq!(
            ToolCallFormat::infer("gpt-oss-120b"),
            ToolCallFormat::GptOss
        );
        assert_eq!(
            ToolCallFormat::infer("something-unknown"),
            ToolCallFormat::Llama3,
            "the fallback is the shape an untrained model improvises"
        );
        // The arm that could not fire: `gemma4` was the only spelling
        // tested, and the checkpoint on hand calls itself
        // `Gemma-4-E2B-It` -- so every gemma tool call fell through to
        // the Llama 3 fallback, which reads none of them. The earlier
        // Gemmas are a different wire format and must still not match.
        for name in ["Gemma-4-E2B-It", "gemma4-27b", "gemma_4_27b", "Gemma 4 27B"] {
            assert_eq!(
                ToolCallFormat::infer(name),
                ToolCallFormat::Gemma4,
                "{name} is a Gemma 4 checkpoint"
            );
        }
        for name in ["gemma-2-2b-it", "gemma-3-12b-it"] {
            assert_ne!(
                ToolCallFormat::infer(name),
                ToolCallFormat::Gemma4,
                "{name} does not write gemma 4's call syntax"
            );
        }
        assert_eq!(ToolCallFormat::parse("qwen"), Some(ToolCallFormat::Qwen25));
        assert_eq!(ToolCallFormat::parse("nonsense"), None);
    }

    #[test]
    fn the_cheap_marker_test_answers_before_any_parsing() {
        assert!(!ToolCallParser::text_may_contain_call("just an answer"));
        assert!(ToolCallParser::text_may_contain_call("<tool_call>{}"));
        assert!(ToolCallParser::text_may_contain_call("[TOOL_CALLS] []"));
    }

    #[test]
    fn balanced_scanning_respects_strings_and_escapes() {
        assert_eq!(json_object_end(r#"{"a": "}"} tail"#), Some(10));
        assert_eq!(json_object_end(r#"{"a": "\""} tail"#), Some(11));
        assert_eq!(json_object_end("{unterminated"), None);
        assert_eq!(json_array_end("[1, [2], 3] tail"), Some(11));
    }
}
