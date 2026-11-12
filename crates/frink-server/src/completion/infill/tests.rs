use std::time::Duration;

use axum::http::StatusCode;
use frink_models::tokenizer::FimTokens;
use serde_json::json;

use super::*;
use crate::model::{GgufFacts, ServerTokenizer};
use crate::response_cache::ResponseCache;
use crate::tests::{post_json_uri, test_app_with_state, test_byte_model_with_facts, test_state};

/// The byte vocabulary: one id per byte, so an assembled prompt can be
/// read back exactly.
fn bytes(text: &str) -> Vec<usize> {
    text.bytes().map(usize::from).collect()
}

const PRE: usize = 250;
const SUF: usize = 251;
const MID: usize = 252;
const REP: usize = 253;
const SEP: usize = 254;

fn fim(rep: bool, sep: bool) -> FimIds {
    FimIds {
        pre: PRE,
        suf: SUF,
        mid: MID,
        rep: rep.then_some(REP),
        sep: sep.then_some(SEP),
    }
}

fn budget(n_batch: i64, n_ctx: Option<i64>, spm: bool) -> InfillBudget {
    InfillBudget {
        n_batch,
        n_predict: SERVER_N_PREDICT,
        n_ctx,
        spm,
    }
}

fn chunk(filename: &str, text: &str) -> ExtraChunk {
    ExtraChunk {
        text: text.to_string(),
        filename: filename.to_string(),
    }
}

fn concat(parts: &[&[usize]]) -> Vec<usize> {
    parts.iter().flat_map(|p| p.iter().copied()).collect()
}

/// `[FIM_PRE] prefix prompt [FIM_SUF] suffix [FIM_MID]`, with the BOS in
/// front when the vocabulary adds one and the prompt at the END OF THE
/// PREFIX, which is where upstream's code puts it.
///
/// Sabotage: append the prompt after `FIM_MID` instead; this goes red.
#[test]
fn the_prefix_suffix_middle_layout_is_upstreams_with_the_prompt_closing_the_prefix() {
    let encode = |t: &str| bytes(t);
    let got = format_infill(
        &encode,
        fim(false, false),
        None,
        bytes("ab"),
        bytes("cd"),
        &[],
        budget(2048, None, false),
        &bytes("p"),
    )
    .unwrap();
    assert_eq!(
        got,
        concat(&[&[PRE], &bytes("abp"), &[SUF], &bytes("cd"), &[MID]])
    );

    let with_bos = format_infill(
        &encode,
        fim(false, false),
        Some(1),
        bytes("ab"),
        bytes("cd"),
        &[],
        budget(2048, None, false),
        &[],
    )
    .unwrap();
    assert_eq!(
        with_bos,
        concat(&[&[1, PRE], &bytes("ab"), &[SUF], &bytes("cd"), &[MID]])
    );
}

/// `--spm-infill` swaps the two halves, the BOS staying first.
#[test]
fn spm_puts_the_suffix_first() {
    let encode = |t: &str| bytes(t);
    let got = format_infill(
        &encode,
        fim(false, false),
        Some(1),
        bytes("ab"),
        bytes("cd"),
        &[],
        budget(2048, None, true),
        &[],
    )
    .unwrap();
    assert_eq!(
        got,
        concat(&[&[1, SUF], &bytes("cd"), &[PRE], &bytes("ab"), &[MID]])
    );
}

/// The repo-level pattern: `[FIM_REP]myproject\n`, each chunk as
/// `[FIM_SEP]filename\n text`, a closing `[FIM_SEP]filename\n`, all of
/// it BEFORE the BOS. Without a file separator a chunk is introduced by
/// upstream's `--- snippet ---` line instead.
#[test]
fn input_extra_is_assembled_as_the_repo_level_pattern_before_the_bos() {
    let encode = |t: &str| bytes(t);
    let extra = [chunk("a.py", "x=1"), chunk("tmp", "y")];
    let got = format_infill(
        &encode,
        fim(true, true),
        Some(1),
        bytes("P"),
        bytes("S"),
        &extra,
        budget(2048, Some(1 << 20), false),
        &[],
    )
    .unwrap();
    assert_eq!(
        got,
        concat(&[
            &[REP],
            &bytes("myproject\n"),
            &[SEP],
            &bytes("a.py\nx=1"),
            &[SEP],
            &bytes("tmp\ny"),
            &[SEP],
            &bytes("filename\n"),
            &[1, PRE],
            &bytes("P"),
            &[SUF],
            &bytes("S"),
            &[MID],
        ])
    );

    let no_sep = format_infill(
        &encode,
        fim(false, false),
        None,
        bytes("P"),
        bytes("S"),
        &[chunk("a.py", "x")],
        budget(2048, Some(1 << 20), false),
        &[],
    )
    .unwrap();
    assert_eq!(
        no_sep,
        concat(&[
            &bytes("\n\n--- snippet ---\n\nx"),
            &[PRE],
            &bytes("P"),
            &[SUF],
            &bytes("S"),
            &[MID],
        ])
    );
}

/// upstream's arithmetic: the prefix keeps its LAST `3 * (n_batch / 4)`
/// tokens, the suffix its FIRST `n_batch / 4 - (2 + prompt)`, and the
/// extra context its last `n_ctx - n_batch - 2 * n_predict`.
///
/// Sabotage: keep the FIRST prefix tokens instead of the last; this goes
/// red.
#[test]
fn the_prefix_suffix_and_extra_budgets_are_upstreams() {
    let encode = |t: &str| bytes(t);
    // n_batch 16: the prefix keeps 12, the suffix 16/4 - (2 + 1) = 1.
    let got = format_infill(
        &encode,
        fim(false, false),
        None,
        bytes("0123456789abcdefgh"),
        bytes("STUVW"),
        &[],
        budget(16, None, false),
        &bytes("p"),
    )
    .unwrap();
    assert_eq!(
        got,
        concat(&[&[PRE], &bytes("6789abcdefghp"), &[SUF], &bytes("S"), &[MID]])
    );

    // n_ctx 20, n_batch 16, n_predict -1: 20 - 16 + 2 = 6 extra tokens,
    // the last six of `[REP]myproject\n`.
    let got = format_infill(
        &encode,
        fim(true, false),
        None,
        Vec::new(),
        Vec::new(),
        &[],
        budget(16, Some(20), false),
        &[],
    )
    .unwrap();
    assert_eq!(got, concat(&[&bytes("oject\n"), &[PRE, SUF, MID]]));
}

/// Extra context with no context size to budget it against is refused
/// rather than guessed; with no extra context the size is not needed.
#[test]
fn extra_context_without_a_context_size_is_the_one_refusal() {
    let encode = |t: &str| bytes(t);
    let run = |fim: FimIds, extra: &[ExtraChunk]| {
        format_infill(
            &encode,
            fim,
            None,
            bytes("a"),
            bytes("b"),
            extra,
            budget(2048, None, false),
            &[],
        )
    };
    assert_eq!(run(fim(true, false), &[]), Err(NoContextSize));
    assert_eq!(
        run(fim(false, false), &[chunk("f", "x")]),
        Err(NoContextSize)
    );
    assert!(run(fim(false, false), &[]).is_ok());
}

/// Each of upstream's body checks, in its words.
#[test]
fn the_body_is_validated_with_upstreams_messages() {
    let message = |body: Value| {
        let (status, body) = InfillFields::parse(&body).err().expect("refused");
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body.0["error"]["type"], "invalid_request_error");
        body.0["error"]["message"].as_str().unwrap().to_string()
    };
    assert_eq!(
        message(json!({"prompt": 3, "input_prefix": "", "input_suffix": ""})),
        "\"prompt\" must be a string"
    );
    assert_eq!(
        message(json!({"input_suffix": ""})),
        "\"input_prefix\" is required"
    );
    assert_eq!(
        message(json!({"input_prefix": ""})),
        "\"input_suffix\" is required"
    );
    assert_eq!(
        message(json!({"input_prefix": "", "input_suffix": "", "input_extra": {}})),
        "\"input_extra\" must be an array of {\"filename\": string, \"text\": string}"
    );
    assert_eq!(
        message(
            json!({"input_prefix": "", "input_suffix": "", "input_extra": [{"filename": "a"}]})
        ),
        "extra_context chunk must contain a \"text\" field with a string value"
    );
    assert_eq!(
        message(
            json!({"input_prefix": "", "input_suffix": "", "input_extra": [{"text": "a", "filename": 1}]})
        ),
        "extra_context chunk's \"filename\" field must be a string"
    );

    let ok = json!({"input_prefix": "a", "input_suffix": "b", "input_extra": [{"text": "t"}]});
    let fields = InfillFields::parse(&ok).expect("valid");
    assert_eq!(fields.prompt, "");
    assert_eq!(fields.input_extra, vec![chunk("tmp", "t")]);
}

/// A prefix is a string, or strings and token ids mixed, every string
/// tokenized without `parse_special`.
#[test]
fn a_prefix_may_mix_strings_and_token_ids() {
    let encode = |t: &str| bytes(t);
    assert_eq!(
        tokenize_mixed(&encode, &json!("ab"), "input_prefix").unwrap(),
        bytes("ab")
    );
    assert_eq!(
        tokenize_mixed(&encode, &json!(["a", 7, "b"]), "input_prefix").unwrap(),
        vec![97, 7, 98]
    );
    for bad in [json!(5), json!([-1]), json!([1.5]), json!([{}])] {
        let (status, _) = tokenize_mixed(&encode, &bad, "input_prefix").unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST, "{bad}");
    }
}

/// The byte-vocabulary fixture with these FIM tokens.
fn app_with(fim: FimTokens) -> axum::Router {
    let model = test_byte_model_with_facts(GgufFacts {
        fim,
        file_type: None,
    });
    test_app_with_state(Arc::new(test_state(
        model,
        ResponseCache::new(16, Duration::from_secs(60)),
    )))
}

fn psm_only() -> FimTokens {
    FimTokens {
        pre: Some(PRE as u32),
        suf: Some(SUF as u32),
        mid: Some(MID as u32),
        ..FimTokens::default()
    }
}

/// A vocabulary without the three FIM tokens is upstream's 501, naming
/// each one missing, before the body is looked at.
///
/// Sabotage: drop the `suffix token is missing. ` line; this goes red.
#[tokio::test]
async fn a_model_without_fim_tokens_is_refused_naming_each_one() {
    let (status, body) = post_json_uri(
        &app_with(FimTokens::default()),
        frink_api::routes::INFILL,
        json!({"input_prefix": "a", "input_suffix": "b"}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert_eq!(body["error"]["type"], "not_supported_error");
    assert_eq!(body["error"]["code"], 501);
    assert_eq!(
        body["error"]["message"],
        "Infill is not supported by this model: prefix token is missing. suffix token is \
         missing. middle token is missing. "
    );

    let only_mid_missing = FimTokens {
        mid: None,
        ..psm_only()
    };
    let (_, body) = post_json_uri(
        &app_with(only_mid_missing),
        frink_api::routes::INFILL,
        json!({}),
    )
    .await;
    assert_eq!(
        body["error"]["message"],
        "Infill is not supported by this model: middle token is missing. "
    );
}

/// The whole route: the assembled ids reach the decode (its
/// `tokens_evaluated`), the echo is their detokenization, and the
/// answer is `/completion`'s object.
#[tokio::test]
async fn an_infill_is_served_as_a_completion_of_the_assembled_prompt() {
    let (status, body) = post_json_uri(
        &app_with(psm_only()),
        frink_api::routes::INFILL,
        json!({
            "input_prefix": "def f(",
            "input_suffix": "):\n",
            "prompt": "x",
            "n_predict": 2,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let expected = concat(&[&[PRE], &bytes("def f(x"), &[SUF], &bytes("):\n"), &[MID]]);
    assert_eq!(body["tokens_evaluated"], expected.len());
    assert_eq!(body["prompt"], ServerTokenizer::Byte.decode(&expected));
    assert_eq!(body["stop"], true);
    assert_eq!(body["id_slot"], -1);
    assert!(body["content"].is_string());
    assert!(body["timings"].is_object());
    assert_eq!(body["generation_settings"]["n_predict"], 2);
}

/// `/completion`'s own refusals still apply: the body is a completion
/// request once its prompt is assembled.
#[tokio::test]
async fn a_completion_option_frink_lacks_is_refused_on_infill_too() {
    let (status, body) = post_json_uri(
        &app_with(psm_only()),
        frink_api::routes::INFILL,
        json!({"input_prefix": "a", "input_suffix": "b", "n_predict": 1, "mirostat": 2}),
    )
    .await;
    assert_eq!(status, StatusCode::NOT_IMPLEMENTED);
    assert!(
        body["error"]["message"]
            .as_str()
            .unwrap()
            .contains("mirostat"),
        "{body}"
    );
}
