use super::*;

/// The invariant behind keying rows by uid: a row leaving the batch
/// must not renumber the rows that stay. A stale id resolves to
/// nothing; a live id still resolves to its *own* state.
#[test]
fn removing_a_row_never_reassigns_another_rows_state() {
    let mut rows = Rows::default();
    let (a, _ra) = test_slot(3, 11);
    let (b, _rb) = test_slot(5, 22);
    let (c, _rc) = test_slot(7, 33);
    let a = rows.insert(a);
    let b = rows.insert(b);
    let c = rows.insert(c);
    assert_eq!(rows.order, vec![a, b, c]);

    let removed = rows.remove(b).expect("b was present");
    assert_eq!(removed.max_tokens, 5);

    assert!(
        rows.get(b).is_none(),
        "a stale uid must resolve to nothing, never to another request's row"
    );
    assert_eq!(rows.get(a).expect("a still in flight").max_tokens, 3);
    assert_eq!(
        rows.get(c).expect("c still in flight").max_tokens,
        7,
        "c must still be c after b left"
    );
    assert_eq!(rows.order, vec![a, c], "admission order is preserved");
    assert_eq!(rows.len(), 2);

    // The positional equivalent, spelled out: `swap_remove` moves
    // the last row into the removed slot, so an index captured for
    // C before the removal now addresses B's old position -- or
    // nothing. Same removal, silently wrong answer.
    let mut positional = vec![3usize, 5, 7];
    let c_index = 2;
    positional.swap_remove(1);
    assert_eq!(positional[1], 7, "C moved into B's index");
    assert!(
        positional.get(c_index).is_none(),
        "C's index now names nothing"
    );
}

/// A new row joining the table must not disturb the rows already in
/// it, and uids are never reused -- so a reply channel and a
/// sampler always travel with the request that owns them.
#[test]
fn uids_are_unique_and_insertion_does_not_disturb_existing_rows() {
    let mut rows = Rows::default();
    let (a, _ra) = test_slot(3, 11);
    let a = rows.insert(a);
    let (b, _rb) = test_slot(5, 22);
    let b = rows.insert(b);
    rows.remove(a);
    let (c, _rc) = test_slot(7, 33);
    let c = rows.insert(c);
    assert_ne!(c, a, "a uid is never reused after its row leaves");
    assert_ne!(c, b);
    assert_eq!(rows.get(b).expect("b untouched").max_tokens, 5);
    assert_eq!(rows.get(c).expect("c inserted").max_tokens, 7);
}

/// `ready` skips finished rows, and `flush_finished` replies to and
/// removes exactly those -- each on its own channel.
#[test]
fn flush_replies_on_each_rows_own_channel() {
    let mut rows = Rows::default();
    let (a, ra) = test_slot(3, 11);
    let (mut b, rb) = test_slot(5, 22);
    b.finish = Some(FinishReason::Stop);
    b.answer.visible.push_str("bee");
    b.answer.generated_ids.push(7);
    let a = rows.insert(a);
    let b = rows.insert(b);
    let (c, _rc) = test_slot(7, 33);
    let c = rows.insert(c);

    assert_eq!(rows.ready(), vec![a, c], "a finished row takes no step");
    rows.flush_finished(&no_budget());
    assert!(rows.get(b).is_none());
    assert_eq!(rows.order, vec![a, c]);

    let (finish, ids, text, usage) =
        finished_result(rb.try_recv().expect("b's caller got a reply")).expect("ok");
    assert_eq!(finish, FinishReason::Stop);
    assert_eq!(ids, vec![7]);
    assert_eq!(text, "bee");
    assert_eq!(usage.completion_tokens, 1);
    assert!(
        ra.try_recv().is_err(),
        "an unfinished row's caller must not be replied to"
    );
}

/// A batched row must report the same RATES the private `generate`
/// loop reports, not just its token counts.
///
/// This is the regression the `RowClock` exists for: continuous
/// batching built its `Usage` with `Usage::new` and never called
/// `with_timings`, so every field Frink Studio puts under an answer
/// came back null -- and because batching is the default on Metal,
/// that was the common case rather than a corner. A unit test on the
/// clock alone would not have caught it: what was missing was the CALL
/// at the reply site, so the assertion has to run through
/// `flush_finished`.
#[test]
fn a_finished_batched_row_reports_its_rates() {
    let mut rows = Rows::default();
    let (mut row, rx) = test_slot(4, 44);
    row.answer.prompt_tokens = 6;
    row.answer.clock.prefill_finished();
    row.answer.clock.token();
    row.answer.generated_ids.push(7);
    row.finish = Some(FinishReason::Stop);
    rows.insert(row);
    rows.flush_finished(&no_budget());

    let (_, _, _, usage) = finished_result(rx.try_recv().expect("replied")).expect("ok");
    assert!(
        usage.prompt_per_second.is_some(),
        "batched row reported no prefill rate: {usage:?}"
    );
    assert!(
        usage.predicted_per_second.is_some(),
        "batched row reported no decode rate: {usage:?}"
    );
    assert!(
        usage.time_to_first_token_ms.is_some(),
        "batched row reported no TTFT: {usage:?}"
    );
}

/// End to end, with the batch mutation that renumbers a positional
/// table: three concurrent rows, one of which trips a stop sequence
/// mid-batch and leaves while the other two keep decoding. In that
/// tick the batch is narrower than the row table, which is exactly
/// when a batch index stops meaning what a row id means. Each
/// caller must still get its own output.
#[test]
fn a_row_leaving_mid_batch_does_not_shift_its_neighbours_output() {
    let decoder = tiny_decoder();
    let prompts = [vec![1usize, 2, 3], vec![4usize, 5], vec![6usize]];
    let budgets = [25usize, 25, 20];
    let refs: Vec<Vec<usize>> = prompts
        .iter()
        .zip(budgets.iter())
        .map(|(p, &n)| sequential_ids(&decoder, p, &greedy_params(n, 4)))
        .collect();

    // The middle row stops on a two-character run from its own
    // stream, so it leaves the batch while its neighbours decode on.
    let letter = |id: &usize| char::from_u32(65 + (*id as u32 % 26)).unwrap_or('?');
    let middle_text: String = refs[1].iter().map(letter).collect();
    assert!(middle_text.len() >= 4);
    let stop = middle_text[2..4].to_string();

    let batcher = ContinuousBatcher::spawn_with_config(
        Arc::clone(&decoder),
        identity_decode(),
        BatcherConfig {
            prefill_chunk: 1,
            ..BatcherConfig::default()
        },
    );
    let barrier = Arc::new(Barrier::new(prompts.len()));
    let handles: Vec<_> = (0..prompts.len())
        .map(|i| {
            let batcher = batcher.clone();
            let barrier = Arc::clone(&barrier);
            let prompt = prompts[i].clone();
            let mut params = greedy_params(budgets[i], 4);
            if i == 1 {
                params.stop = vec![stop.clone()];
            }
            thread::spawn(move || {
                barrier.wait();
                batcher
                    .generate(prompt, params, StopTokens::default())
                    .expect("generate")
                    .1
            })
        })
        .collect();
    let got: Vec<Vec<usize>> = handles.into_iter().map(|h| h.join().unwrap()).collect();

    assert_eq!(got[0], refs[0], "row 0 received another row's output");
    assert_eq!(got[2], refs[2], "row 2 received another row's output");
    assert!(
        got[1].len() < refs[1].len() && refs[1].starts_with(&got[1]),
        "the stopped row must be a strict prefix of its own stream"
    );
}
