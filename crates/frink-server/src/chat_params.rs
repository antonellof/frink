//! What one `/v1/chat/completions` body resolves to before anything
//! decodes: the prompt it renders, and the [`GenerationParams`] it runs
//! under.
//!
//! Split out of `lib.rs` under the "a new file beats a new section"
//! rule when the reasoning budget and the continuation default were
//! added, because both are properties of exactly this seam: the prompt
//! is where a trailing assistant message is continued (one rule for
//! every chat-shaped route), and the params are where the budget is
//! carried, refused for a family that cannot honour it, and keyed for
//! the response cache. The methods are `pub(crate)` so `/v1/messages`
//! and `/v1/responses`, which lower to the same request type, resolve
//! through the same functions rather than their own copies.

use crate::chat_template;
use crate::continuation;
use crate::generate::GenerationParams;
use crate::policy;
use crate::reasoning_budget;
use crate::response_cache::{self, CacheKey};
use crate::tool_grammar;
use crate::{
    grammar_request, prompt_from_messages, unsupported_feature, ApiError, ChatCompletionRequest,
    ChatMessage, ToolDef,
};

impl ChatCompletionRequest {
    /// The prompt this request decodes from: the history rendered
    /// whole, or continued from its last message -- when the caller
    /// asked for that by name, or by the server default when the last
    /// message is an assistant turn (`ContinueFinalMessage::resolve`).
    /// ONE call for every chat-shaped route -- both chat handlers,
    /// `/v1/messages`, `/v1/responses` and `count_tokens` -- so no two
    /// of them can disagree about what a trailing assistant message
    /// means.
    ///
    /// `tools` is a parameter rather than `self.tools` because the
    /// Responses route offers the template a narrowed list.
    /// `served_model` is the name `OutputPosture::resolve` will read the
    /// output with, so the continuation is written in the same family's
    /// markers the parser will look for.
    pub(crate) fn render_prompt(
        &self,
        history: &[ChatMessage],
        template: &chat_template::PromptTemplate,
        tools: &[ToolDef],
        kwargs: serde_json::Map<String, serde_json::Value>,
        served_model: &str,
    ) -> Result<String, ApiError> {
        match self.continue_final_message.resolve(history) {
            None => prompt_from_messages(history, template, tools, kwargs),
            Some((mode, implied)) => continuation::prompt_continuing_final_message(
                history,
                template,
                tools,
                kwargs,
                template.reasoning_format(served_model),
                mode,
            )
            .map_err(|e| {
                if implied {
                    continuation::implied_by_default(e)
                } else {
                    e
                }
            }),
        }
    }

    /// Fallible because a constraint is compiled here: an unparseable
    /// grammar, or a `response_format` this server cannot honour, is a
    /// refusal rather than a request served without the constraint it
    /// asked for.
    pub(crate) fn generation_params(
        &self,
        model: crate::sampling_knobs::SamplerModel<'_>,
    ) -> Result<GenerationParams, ApiError> {
        Ok(GenerationParams {
            // The prompt is prefilled once and the KV forked per
            // choice (`crate::generate`). A STREAMING request never
            // reaches here with more than 1: `chat_completions_stream`
            // refuses the pair by name.
            // `best_of` decides how many are GENERATED, `n` how many
            // come back (`crate::best_of`).
            n: self.unimplemented.candidates(),
            // Reporting costs the greedy fast path, so only a request
            // that will render them asks for them.
            wants_logprobs: self.n_logprobs().is_ok_and(|n| n.is_some())
                || self.unimplemented.ranks_candidates(),
            // Set by `generation_params_for_template`, which is the only
            // caller that knows the SERVED model name. Left `None` here
            // so a path that never resolves it reports the field absent
            // rather than claiming the model did not think.
            reasoning: None,
            max_tokens: self.max_tokens,
            sampling: self.sampling_params(model)?,
            seed: self.resolved_seed(),
            stop: self.effective_stop_sequences(),
            // Resolved by `run_generation_emit`, the layer that holds a
            // tokenizer: a request body names stop strings, and only
            // the model can say which of them are single tokens.
            stop_token_ids: Vec::new(),
            json_object: self.json_object_mode(),
            grammar: grammar_request::for_request(
                self.grammar.as_deref(),
                self.response_format.as_ref(),
            )?,
            // Filled in by the handler that owns the request id --
            // the request body cannot name its own cancel token.
            cancel: None,
            ignore_eos: self.ignore_eos.unwrap_or(false),
            // The number only; it is tokenized against the checkpoint
            // at `run_generation_emit`, beside `stop_token_ids`.
            reasoning_budget: reasoning_budget::ReasoningBudget::from_tokens(
                reasoning_budget::BudgetTokens::effective(self.reasoning_budget_tokens),
            ),
            // Resolved by the handler against the loaded adapters
            // (`crate::lora::resolve_request`), which the request body
            // cannot see.
            lora: None,
        })
    }

    /// Like [`Self::generation_params`], plus architecture-default stop
    /// strings (Gemma IT emits `<end_of_turn>` before `<eos>`) and, for a
    /// forced `tool_choice`, the grammar that makes it forced.
    ///
    /// `served_model` is the name of the checkpoint this generation will
    /// actually run against -- `active.name()`, the same string
    /// [`output::OutputPosture::resolve`] reads the answer back with, and
    /// NOT the `model` field of the request. The two can differ, and a
    /// grammar built for one wire format while the response is parsed in
    /// another would force a call this server then cannot read.
    pub(crate) fn generation_params_for_template(
        &self,
        template: &chat_template::PromptTemplate,
        served_model: &str,
        model: crate::sampling_knobs::SamplerModel<'_>,
    ) -> Result<GenerationParams, ApiError> {
        let mut params = self.generation_params(model)?;
        // The served model, not the request's `model` field -- see this
        // function's doc. Same name `OutputPosture::resolve` reads the
        // answer back with, so the count and the split cannot disagree
        // about which family this checkpoint is.
        params.reasoning = template.reasoning_format(served_model);
        // A budget the served family cannot honour is a 501 here, before
        // any prompt is rendered, and by name: the same rule the
        // tokenizer seam applies, so the two cannot disagree about
        // which families those are.
        if params.reasoning_budget.needs_vocab_logits() {
            if let Some(why) = reasoning_budget::ReasoningBudget::unsupported_for(params.reasoning)
            {
                return Err(unsupported_feature(&why));
            }
        }
        if let Some(stop) = template.end_of_turn() {
            if !params.stop.iter().any(|s| s == stop) {
                params.stop.push(stop.to_string());
            }
        }
        if let Some(forced) = self.forced_tool_choice()? {
            // `validate_supported_fields` has already refused the
            // combinations that would put two constraints on one
            // generation, so there is nothing here to overwrite.
            params.grammar = Some(tool_grammar::build(
                forced,
                &self.tool_specs(),
                policy::parser::ToolCallFormat::infer(served_model),
            )?);
        }
        Ok(params)
    }

    /// A request only has a deterministic outcome -- and therefore is
    /// only safe to serve from or populate into the whole-response
    /// cache -- when it's plain greedy decode (temperature <= 0) or an
    /// explicit seed was given. Anything else must always regenerate:
    /// a "cache hit" for an unseeded sampled request would silently
    /// replay one random draw forever, defeating the purpose of
    /// sampling and surprising any client expecting fresh output per
    /// call.
    pub(crate) fn is_cacheable(&self) -> bool {
        // A request that asked for logprobs must MISS and must not
        // store: `CachedCompletion` holds text and finish reasons, not
        // distributions, so replaying an entry for it would answer a
        // logprobs request with no logprobs and a 200 -- the
        // silent-wrong class. Making it uncacheable is the honest
        // answer while the entry cannot carry them; storing them is a
        // row of its own, and the note is on `CachedCompletion`.
        //
        // `n_logprobs` rather than the raw fields, so this and the
        // renderer cannot disagree about what "asked for logprobs"
        // means.
        if self.n_logprobs().is_ok_and(|n| n.is_some()) {
            return false;
        }
        self.temperature.unwrap_or(0.0) <= 0.0 || self.seed.is_some()
    }

    /// The cache key for this request under the parameters it will
    /// actually be generated with.
    ///
    /// `params` is taken rather than rebuilt because the RESOLVED
    /// parameters are the only honest thing to key on: this function
    /// used to re-state a handful of the request's fields, complete with
    /// its own copy of every `unwrap_or` default, and then keyed on a
    /// configuration that was only nearly the one that ran. Three fields
    /// of that hand-written list were simply missing (#35).
    ///
    /// `params` must be the ones from
    /// [`Self::generation_params_for_template`], not
    /// [`Self::generation_params`]: the template's end-of-turn stop and
    /// a forced `tool_choice`'s grammar are added there, and both change
    /// the answer.
    pub(crate) fn cache_key(&self, prompt: &str, params: &GenerationParams) -> CacheKey {
        CacheKey {
            model: self.model.clone(),
            prompt: prompt.to_string(),
            generation: response_cache::generation_key(params),
            seed: self.seed,
        }
    }

    pub(crate) fn resolved_seed(&self) -> u64 {
        self.seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(0xDEFA017)
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::chat_template;
    use crate::ChatCompletionRequest;
    use axum::http::StatusCode;

    fn chat_request(value: serde_json::Value) -> ChatCompletionRequest {
        serde_json::from_value(value).expect("request")
    }

    /// The reasoning split is resolved from the SERVED model, and it is
    /// what decides whether `usage.completion_tokens_details` exists at
    /// all. Resolved from the request's `model` field instead, a client
    /// naming an alias would silently get no count -- and `None` here is
    /// indistinguishable on the wire from "this model did not think",
    /// which is the confusion #120 is about.
    #[test]
    fn the_reasoning_split_is_resolved_from_the_served_model_not_the_request() {
        let req = chat_request(serde_json::json!({
            // Deliberately a name that infers NOTHING, so a pass can only
            // come from the served name below.
            "model": "some-alias",
            "messages": [{"role": "user", "content": "hi"}],
        }));
        let template = chat_template::PromptTemplate::plain();

        let thinks = req
            .generation_params_for_template(
                &template,
                "Qwen3-8B",
                crate::sampling_knobs::SamplerModel::absent(),
            )
            .expect("params");
        assert!(
            thinks.reasoning.is_some(),
            "a thinking checkpoint must carry its format into generation"
        );

        let plain = req
            .generation_params_for_template(
                &template,
                "Llama-3.2-1B-Instruct",
                crate::sampling_knobs::SamplerModel::absent(),
            )
            .expect("params");
        assert!(
            plain.reasoning.is_none(),
            "a checkpoint with no reasoning format must carry none, so the \
             usage field stays absent rather than becoming a zero"
        );
    }

    /// The request-level half of `continuation`: the field reaches the
    /// render, and the family is taken from the SERVED model, the same
    /// name the output parser reads. A trailing assistant turn WITHOUT
    /// the field is continued too -- llama.cpp's server default
    /// (`server-common.cpp:1046-1056`) -- and `false` is the one way a
    /// request renders it as history plus a fresh turn.
    #[test]
    fn continue_final_message_reaches_the_render_under_the_served_models_family() {
        let r1 = chat_template::PromptTemplate::from_gguf_metadata(
            Some("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|><think>\n{% endif %}"),
            Some("qwen2"),
            false,
            true,
            None,
            None,
        );
        let body = serde_json::json!({
            "model": "m",
            "messages": [
                {"role": "user", "content": "why"},
                {"role": "assistant", "content": "", "reasoning_content": "Let me"},
            ],
        });
        let mut closed = body.clone();
        closed["continue_final_message"] = serde_json::json!(false);
        let plain = chat_request(closed);
        let prompt = plain
            .render_prompt(
                &plain.messages,
                &r1,
                &plain.tools,
                serde_json::Map::new(),
                "DeepSeek-R1-Distill",
            )
            .expect("renders");
        assert_eq!(prompt, "<|user|>why<|assistant|><|assistant|><think>\n");

        let implied = chat_request(body.clone());
        let prompt = implied
            .render_prompt(
                &implied.messages,
                &r1,
                &implied.tools,
                serde_json::Map::new(),
                "DeepSeek-R1-Distill",
            )
            .expect("renders");
        assert_eq!(
            prompt, "<|user|>why<|assistant|><think>Let me",
            "a trailing assistant message is continued by default"
        );

        let mut continued = body;
        continued["continue_final_message"] = serde_json::json!(true);
        let req = chat_request(continued);
        let prompt = req
            .render_prompt(
                &req.messages,
                &r1,
                &req.tools,
                serde_json::Map::new(),
                "DeepSeek-R1-Distill",
            )
            .expect("renders");
        assert_eq!(prompt, "<|user|>why<|assistant|><think>Let me");
        // Under a served model with no reasoning family the same body
        // is a refusal, not a guess. The family comes from the name OR
        // the template, so the template here must be silent about
        // thinking too: `r1`'s opens `<think>` itself and would (rightly)
        // imply the parser for any name.
        let bare = chat_template::PromptTemplate::from_gguf_metadata(
            Some("{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"),
            Some("llama"),
            false,
            true,
            None,
            None,
        );
        let (status, _) = req
            .render_prompt(
                &req.messages,
                &bare,
                &req.tools,
                serde_json::Map::new(),
                "Llama-3.2-3B",
            )
            .expect_err("no family to write the thought in");
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        // The same refusal reached by DEFAULT names the way out, because
        // the caller never asked for a continuation by name.
        let (status, body) = implied
            .render_prompt(
                &implied.messages,
                &bare,
                &implied.tools,
                serde_json::Map::new(),
                "Llama-3.2-3B",
            )
            .expect_err("no family to write the thought in");
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert!(body.0["error"]["message"]
            .as_str()
            .unwrap()
            .contains("--no-prefill-assistant"));
    }

    /// llama.cpp's budget field, under both of its spellings, reaches
    /// the generation as a `Requested` number: the tokenizer seam turns
    /// it into a plan, and the sampler refuses to run without one.
    /// `-1` and absence are `Unrestricted`, which builds no machine
    /// (`common/sampling.cpp:316`). An out-of-range value is a 400 at
    /// deserialization, naming the field (`arg.cpp:3611`,
    /// `server-schema.cpp:384`).
    #[test]
    fn a_reasoning_budget_reaches_the_generation_under_both_spellings() {
        use crate::reasoning_budget::ReasoningBudget;
        let template = chat_template::PromptTemplate::plain();
        for key in ["reasoning_budget_tokens", "thinking_budget_tokens"] {
            let req = chat_request(serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                key: 2000,
            }));
            req.validate_supported_fields().expect("accepted");
            let params = req
                .generation_params_for_template(
                    &template,
                    "DeepSeek-R1-Distill-Qwen-1.5B",
                    crate::sampling_knobs::SamplerModel::absent(),
                )
                .expect("params");
            assert!(
                matches!(params.reasoning_budget, ReasoningBudget::Requested(2000)),
                "{key}: {:?}",
                params.reasoning_budget
            );
            assert!(
                params.needs_vocab_logits(),
                "{key}: forcing the closer needs every logit, so the device argmax fold is off"
            );
        }
        for value in [None, Some(-1)] {
            let mut body = serde_json::json!({
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            });
            if let Some(v) = value {
                body["reasoning_budget_tokens"] = serde_json::json!(v);
            }
            let req = chat_request(body);
            let params = req
                .generation_params_for_template(
                    &template,
                    "DeepSeek-R1-Distill-Qwen-1.5B",
                    crate::sampling_knobs::SamplerModel::absent(),
                )
                .expect("params");
            assert!(
                matches!(params.reasoning_budget, ReasoningBudget::Unrestricted),
                "{value:?}"
            );
        }
        let Err(err) = serde_json::from_value::<ChatCompletionRequest>(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_budget_tokens": -2,
        })) else {
            panic!("below -1 is out of llama.cpp's range");
        };
        assert!(err.to_string().contains("reasoning_budget_tokens"), "{err}");
    }

    /// A budget on a channel-grammar family is a 501 by name before any
    /// prompt is rendered; on a plain checkpoint it is vacuous
    /// (llama.cpp builds no sampler without thinking tags,
    /// `server-common.cpp:1134`).
    #[test]
    fn a_reasoning_budget_on_a_channel_grammar_is_refused_before_rendering() {
        let template = chat_template::PromptTemplate::plain();
        let req = chat_request(serde_json::json!({
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "reasoning_budget_tokens": 0,
        }));
        let Err((status, body)) = req.generation_params_for_template(
            &template,
            "gpt-oss-20b",
            crate::sampling_knobs::SamplerModel::absent(),
        ) else {
            panic!("harmony has no closer to force");
        };
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
        assert!(body.0["error"]["message"]
            .as_str()
            .unwrap()
            .contains("reasoning_budget_tokens"));
        req.generation_params_for_template(
            &template,
            "Llama-3.2-1B-Instruct",
            crate::sampling_knobs::SamplerModel::absent(),
        )
        .expect("no thought to bound");
    }
}
