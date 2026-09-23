use super::*;

/// The timing rule the snapshot type exists for, asserted where it
/// is actually produced.
///
/// `#new-token` must be the prompt tokens this batch is about to
/// compute, counted AT ADMISSION. Read back off the live rows after
/// the forward, every admitted prompt's processed length has caught
/// up to its device length, so the same line would report one token
/// per request -- the decode state of a prefill batch -- and a
/// prefill log becomes useless for the one thing it is read for.
#[test]
fn a_prefill_line_counts_the_tokens_the_batch_is_about_to_compute() {
    let decoder = tiny_decoder();
    let mut waiting: VecDeque<Job> = VecDeque::new();
    let mut prefills: VecDeque<Prefill> = VecDeque::new();
    let queue = QueueGate::new(8);
    let budget = no_budget();
    let mut replies = Vec::new();

    for prompt in [vec![1usize, 2, 3, 4], vec![5usize, 6]] {
        let (tx, rx) = mpsc::channel();
        replies.push(rx);
        queue.try_reserve().expect("room in the queue");
        waiting.push_back(Job {
            prompt_tokens: prompt.clone(),
            params: greedy_params(4, 1),
            stop_tokens: StopTokens::default(),
            reply: tx,
            abort: AbortId(0),
            blocks: 1,
            skips: 0,
        });
    }

    let snapshot = admit(
        &decoder,
        &mut waiting,
        &mut prefills,
        0,
        &BatcherConfig::default(),
        &queue,
        &budget,
        None,
    );
    assert_eq!(snapshot.new_seqs, 2);
    assert_eq!(
        snapshot.new_tokens, 6,
        "4 + 2 prompt tokens, not one per request"
    );

    // And a tick with nothing to admit reports nothing, rather than
    // an all-zero prefill line on every idle pass.
    let idle = admit(
        &decoder,
        &mut waiting,
        &mut prefills,
        0,
        &BatcherConfig::default(),
        &queue,
        &budget,
        None,
    );
    assert_eq!(idle, PrefillSnapshot::default());
}

/// A pool the deployment does not have is `None`, so the reporter
/// omits it entirely. Printed as a row of zeros instead, an
/// operator sizes a window pool this server never had.
#[test]
fn a_status_line_omits_the_pools_this_model_does_not_have() {
    let rows = Rows::default();
    let prefills = VecDeque::new();
    let waiting = VecDeque::new();

    let status = batch_status(&rows, &prefills, &waiting, &budget(16, Some(64)));
    assert!(status.window.is_none());
    assert!(status.recurrent.is_none());
    assert_eq!(status.kv_pages.total, 64);
    assert_eq!(status.kv_pages.used, 0, "an untouched budget holds nothing");

    // With no block budget at all there is no pool to report, and
    // an empty gauge reads as idle rather than as full.
    let status = batch_status(&rows, &prefills, &waiting, &no_budget());
    assert_eq!(status.kv_pages, PoolUsage::default());
}
