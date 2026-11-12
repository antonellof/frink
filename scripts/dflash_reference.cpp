// libllama's answer for DFlash drafting rounds, the golden that
// `crates/frink-models/tests/dflash_graphs.rs` holds frink to.
//
//   dflash_reference <target.gguf> <draft.gguf> <n_max> <rounds> <tok0> <tok1> ...
//
// The loop is `common_speculative_impl_draft_dflash` with greedy
// verification, written out against the public and `llama-ext.h` API so
// every step frink mirrors is visible here:
//
//   1. the target decodes the prompt with its `target_layers` inputs
//      extracted (`llama_set_embeddings_layer_inp`);
//   2. those rows are injected into the draft as embeddings at their
//      positions (`process()`, an embd batch through `fc`);
//   3. the draft decodes `[anchor, mask * n_max]` at the next positions
//      and the DFlash2 lattice is walked greedily (`draft()`, p_min 0),
//      or plain DFlash reads each mask row's argmax;
//   4. the block's KV is removed from the draft, the target verifies
//      `[anchor, drafts...]`, the accepted prefix is committed, and the
//      target's rows for it are what step 2 injects next round.
//
// Printed, one line per round:
//   round <r> anchor <id> draft <id...> scores <f32...>
// where `scores` are the walked lattice scores (DFlash2) or the drafted
// logits (DFlash), one per drafted token.
//
// F32 KV and no flash attention, as every other golden here
// (`gptoss_reference_logits.cpp`).

#include "llama.h"
#include "../src/llama-ext.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static llama_context * make_ctx(llama_model * model, llama_context * other) {
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.n_batch = 128;
    cp.n_ubatch = 128;
    cp.no_perf = true;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.ctx_other = other;
    return llama_init_from_model(model, cp);
}

static int argmax(const float * v, int n) {
    int best = 0;
    for (int i = 1; i < n; ++i) {
        if (v[i] > v[best]) best = i;
    }
    return best;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <target.gguf> <draft.gguf> <n_max> <rounds> <tok...>\n", argv[0]);
        return 2;
    }
    const int n_max_req = atoi(argv[3]);
    const int rounds = atoi(argv[4]);
    std::vector<llama_token> prompt;
    for (int i = 5; i < argc; ++i) prompt.push_back((llama_token) atoi(argv[i]));

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    llama_model * tgt = llama_model_load_from_file(argv[1], mp);
    llama_model * dft = llama_model_load_from_file(argv[2], mp);
    if (!tgt || !dft) {
        fprintf(stderr, "failed to load models\n");
        return 1;
    }
    llama_context * ctx_t = make_ctx(tgt, nullptr);
    llama_context * ctx_d = make_ctx(dft, ctx_t);
    if (!ctx_t || !ctx_d) {
        fprintf(stderr, "failed to create contexts\n");
        return 1;
    }

    const int32_t * layers = llama_model_target_layer_ids(dft);
    const int n_layers = (int) llama_model_target_layer_ids_n(dft);
    for (int k = 0; k < n_layers; ++k) llama_set_embeddings_layer_inp(ctx_t, (uint32_t) layers[k], true);

    const bool dflash2 = llama_model_dflash_selector_top_k(dft) > 0;
    const int top_k = llama_model_dflash_selector_top_k(dft);
    char buf[32] = {};
    int block_size = 16;
    if (llama_model_meta_val_str(dft, "dflash.block_size", buf, sizeof(buf)) >= 0) block_size = atoi(buf);
    bool causal = false;
    if (llama_model_meta_val_str(dft, "dflash.attention.causal", buf, sizeof(buf)) >= 0) causal = strcmp(buf, "true") == 0;
    llama_set_embeddings_nextn(ctx_d, true, !dflash2);
    llama_set_causal_attn(ctx_d, causal);
    const int n_max = std::min(n_max_req, block_size - 1);

    const llama_vocab * vocab = llama_model_get_vocab(tgt);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token mask = llama_vocab_mask(llama_model_get_vocab(dft));
    const int n_embd_t = llama_model_n_embd(tgt);
    const int n_embd_d = llama_model_n_embd(dft);
    const int n_enc = n_layers * n_embd_t;

    llama_batch_ext * bt = llama_batch_ext_init(ctx_t);
    llama_batch_ext * bd = llama_batch_ext_init(ctx_d);

    // Target forward over `toks` at `pos0`, every row an output.
    auto target_forward = [&](const std::vector<llama_token> & toks, llama_pos pos0) {
        llama_batch_ext_clear(bt);
        for (size_t i = 0; i < toks.size(); ++i) {
            const int32_t idx = llama_batch_ext_add_token(bt, 0, toks[i]);
            const llama_pos p = pos0 + (llama_pos) i;
            llama_batch_ext_set_pos(bt, idx, &p);
            llama_batch_ext_set_output_logits(bt, idx, true);
        }
        if (llama_process(ctx_t, LLAMA_PROCESS_TYPE_DECODE, bt) != 0) {
            fprintf(stderr, "target decode failed\n");
            exit(1);
        }
    };
    // Inject the target rows [0, n) of the last target batch at pos0.
    std::vector<float> feats;
    auto inject = [&](int n, llama_pos pos0) {
        feats.assign((size_t) n * n_enc, 0.0f);
        for (int k = 0; k < n_layers; ++k) {
            const float * src = llama_get_embeddings_layer_inp(ctx_t, (uint32_t) layers[k]);
            for (int i = 0; i < n; ++i) {
                memcpy(feats.data() + (size_t) i * n_enc + (size_t) k * n_embd_t,
                       src + (size_t) i * n_embd_t, sizeof(float) * n_embd_t);
            }
        }
        llama_memory_seq_rm(llama_get_memory(ctx_d), 0, pos0, -1);
        llama_batch_ext_clear(bd);
        for (int i = 0; i < n; ++i) {
            const int32_t idx = llama_batch_ext_add_embd(bd, 0, { feats.data() + (size_t) i * n_enc, 1, (size_t) n_enc });
            const llama_pos p = pos0 + i;
            llama_batch_ext_set_pos(bd, idx, &p);
        }
        if (llama_process(ctx_d, LLAMA_PROCESS_TYPE_DECODE, bd) != 0) {
            fprintf(stderr, "draft injection failed\n");
            exit(1);
        }
    };

    target_forward(prompt, 0);
    llama_pos committed = (llama_pos) prompt.size();
    inject((int) prompt.size(), 0);
    llama_token anchor = argmax(llama_get_logits_ith(ctx_t, (int32_t) prompt.size() - 1), n_vocab);

    for (int r = 0; r < rounds; ++r) {
        // Draft [anchor, mask * n_max] at `committed`.
        const int n_block = n_max + 1;
        llama_batch_ext_clear(bd);
        for (int i = 0; i < n_block; ++i) {
            const int32_t idx = llama_batch_ext_add_token(bd, 0, i == 0 ? anchor : mask);
            const llama_pos p = committed + i;
            llama_batch_ext_set_pos(bd, idx, &p);
            llama_batch_ext_set_output_logits(bd, idx, !dflash2);
        }
        if (llama_process(ctx_d, LLAMA_PROCESS_TYPE_DECODE, bd) != 0) {
            fprintf(stderr, "draft decode failed\n");
            return 1;
        }
        std::vector<llama_token> draft;
        std::vector<float> scores;
        if (dflash2) {
            const float * lattice = llama_get_embeddings_nextn(ctx_d);
            int pred = 0;
            for (int i = 1; i < n_block; ++i) {
                const float * row = lattice + (size_t) i * n_embd_d;
                const float * s = row + top_k + (size_t) pred * top_k;
                pred = (int) (std::max_element(s, s + top_k) - s);
                draft.push_back((llama_token) row[pred]);
                scores.push_back(s[pred]);
            }
        } else {
            for (int i = 1; i < n_block; ++i) {
                const float * lg = llama_get_logits_ith(ctx_d, i);
                const int best = argmax(lg, n_vocab);
                draft.push_back(best);
                scores.push_back(lg[best]);
            }
        }
        printf("round %d anchor %d draft", r, anchor);
        for (auto t : draft) printf(" %d", t);
        printf(" scores");
        for (auto s : scores) printf(" %.9g", s);
        printf("\n");

        // The block's KV goes; the target verifies [anchor, draft...].
        llama_memory_seq_rm(llama_get_memory(ctx_d), 0, committed, -1);
        std::vector<llama_token> verify = {anchor};
        verify.insert(verify.end(), draft.begin(), draft.end());
        target_forward(verify, committed);
        int accepted = 0;
        while (accepted < (int) draft.size() &&
               argmax(llama_get_logits_ith(ctx_t, accepted), n_vocab) == draft[accepted]) {
            ++accepted;
        }
        llama_token next = argmax(llama_get_logits_ith(ctx_t, accepted), n_vocab);
        llama_memory_seq_rm(llama_get_memory(ctx_t), 0, committed + 1 + accepted, -1);
        inject(1 + accepted, committed);
        committed += 1 + accepted;
        anchor = next;
    }

    llama_batch_ext_free(bt);
    llama_batch_ext_free(bd);
    llama_free(ctx_d);
    llama_free(ctx_t);
    llama_model_free(dft);
    llama_model_free(tgt);
    llama_backend_free();
    return 0;
}
