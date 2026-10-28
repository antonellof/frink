// Reference dumper for `frink parity`: llama.cpp's own answers for a
// GGUF, in two modes.
//
//   logits    llama.cpp's last-position logits for an EXPLICIT token-id
//             sequence, written as raw f32.
//   tokenize  llama.cpp's token ids for a batch of raw texts, written as
//             length-prefixed i32 runs.
//   embed     llama.cpp's POOLED EMBEDDING for one raw text, printed as
//             one f32 per line.
//
// Why the logits mode takes token ids and not text: `frink` and
// llama.cpp were once compared by greedy text, and it proved nothing —
// the two tokenizers had to agree first, and once they did the sequences
// still diverged after ~3 tokens from ordinary numeric drift flipping a
// single argmax. Text is the wrong medium. That mode dumps the logit
// vector at one position, so the comparison is over a distribution
// instead of over one sampled draw, and the tokenizer is removed from
// the experiment entirely by passing the ids in.
//
// Why the tokenize mode exists: removing the tokenizer from the logits
// experiment left it outside the only cross-engine oracle in the repo,
// and a real defect lived there undetected for the life of the project
// (one hardcoded pre-tokenizer regex for every BPE checkpoint, so every
// run of 4+ digits and every run of 2+ whitespace tokenized differently
// from llama.cpp). The tokenizer needs its OWN oracle, on the same file,
// against the same library. That is this mode.
//
// Build (macOS, Homebrew llama.cpp):
//   ./tools/build_llama_logits.sh
//
// or by hand (macOS, Homebrew llama.cpp):
//   P=$(brew --prefix llama.cpp)
//   cc -std=c11 -O2 -I"$P/include" -L"$P/lib" -lllama \
//       -Wl,-rpath,"$P/lib" tools/llama_logits.c -o target/llama_logits
//
// Why the embed mode exists: an encoder has no "first-token logit", so
// the logits mode cannot reach a BGE/E5/nomic checkpoint at all. What it
// has instead is one pooled vector per sequence, and that is what this
// mode dumps — through llama.cpp's own pooling (`llama_get_embeddings_seq`
// with the model's own `pooling_type`), so the comparison covers the
// pooling rule and not only the graph. It takes TEXT rather than ids,
// unlike the logits mode, because a BERT checkpoint's `[CLS]`/`[SEP]`
// wrapping is part of what is being checked; frink's tokenizer is
// already held to this same library by the tokenize mode above, so the
// tokenizer is not an uncontrolled variable here.
//
// Usage:
//   llama_logits <model.gguf> <out.bin> <tok0> <tok1> ...
//       Writes n_vocab f32 to <out.bin> and prints "libllama <path>"
//       followed by "n_vocab <N>". The path is the library that
//       actually answered — see print_reference_identity below for why
//       a verdict without it is only half an experiment.
//
//       With LLAMA_LOGITS_DECODE_STEPS=N (N >= 1) in the environment it
//       is a GREEDY DECODE instead: the prompt as one batch, then the
//       argmax fed back N-1 times, each as its own one-token
//       llama_decode, and <out.bin> holds N rows of n_vocab f32. Row i
//       is the distribution whose argmax (first maximum) is fed at step
//       i+1. `frink parity --decode-steps` reads it: prefill parity
//       alone cannot see a defect that only shows once decode starts.
//       FRINK_CTK=f32 runs an f32 K/V cache, as frink then does too.
//
//   llama_logits --tokenize <model.gguf> <cases.bin> <out.bin>
//       Reads an FXTK case file, writes an FXTK result file, and prints
//       "tokenized <N> cases".
//
//   llama_logits --embed <model.gguf> <text>
//       Prints n_embd floats, one per line, un-normalized, and prints
//       the token ids llama.cpp used to stderr.
//
// Exit codes: 0 ok, 1 failed, 2 bad arguments or a malformed case file,
// 3 llama.cpp cannot load this checkpoint (see EXIT_MODEL_UNSUPPORTED).
//
// FXTK wire format (little-endian throughout, written byte by byte so
// the two sides do not have to share a host endianness):
//
//   cases   "FXTK" | u32 n_cases | repeat( u32 n_bytes | n_bytes text )
//   result  "FXTK" | u32 version=2 | u32 flags | u32 n_vocab
//                  | u32 n_cases  | repeat( run(parse_special=false)
//                                           run(parse_special=true) )
//           run    u32 n_tokens | n_tokens i32
//           flags bit0 = add_bos, bit1 = add_eos, as the vocab reports
//           them. The ids themselves are dumped with add_special=false,
//           so the BOS policy is compared as a FLAG rather than being
//           baked into every sequence, where one disagreement would
//           misreport every case as a tokenizer divergence.
//
//           Every case is tokenized TWICE, once per parse_special
//           setting, because the two are different questions and
//           frink has to answer both. Version 1 dumped only the
//           parse_special=true answer, and that hid a real defect:
//           frink parsed special-token markers unconditionally, so a
//           document that MENTIONED `<|im_end|>` in prose was off by
//           one token against llama.cpp's default, and the only
//           reference the oracle had was the one setting on which the
//           two agreed.
//
// CPU only (n_gpu_layers = 0) in the logits mode: the frink side of the
// comparison is its CPU path, which is the one cross-validated against
// NumPy. Comparing a GPU reference against a CPU candidate would
// confound two differences. The tokenize mode loads vocab_only, so it
// touches no backend at all.

// dladdr is a GNU extension on glibc; harmless elsewhere.
#define _GNU_SOURCE

#include "llama.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__APPLE__) || defined(__linux__)
#  include <dlfcn.h>
#  define FX_HAVE_DLADDR 1
#endif

#define FXTK_MAGIC "FXTK"
#define FXTK_VERSION 2u

// Exit code for "llama.cpp itself cannot load this checkpoint". Kept
// distinct from the generic failure code because it is not evidence
// about frink: a reference with no answer has no verdict, and a caller
// that cannot tell the two apart has to either ignore real failures or
// report a missing reference as a defect.
#define EXIT_MODEL_UNSUPPORTED 3

// Prints WHICH libllama actually answered, as `libllama <path>`.
//
// A parity verdict is not a property of frink alone: the same
// checkpoint scored DRIFT against one libllama and WRONG against
// another, and the printed report said nothing that distinguished the
// two runs (issue #102). The two builds turned out to differ in the
// K-quant path -- their Q8_0 answers are bit-identical, their Q4_K/Q6_K
// answers are not -- so a report that omits the reference's identity is
// omitting half the experiment.
//
// The path is taken from the LOADED image rather than from the compile
// -time -L flag, because an rpath, a DYLD_LIBRARY_PATH or a Homebrew
// relink can all make those two different, and the one that computed
// the logits is the one that matters.
static void print_reference_identity(void) {
#ifdef FX_HAVE_DLADDR
    Dl_info info;
    // Any exported llama symbol identifies the image; this one exists in
    // every version the dumper can be built against.
    if (dladdr((void *) (uintptr_t) &llama_backend_init, &info) != 0 && info.dli_fname) {
        printf("libllama %s\n", info.dli_fname);
        return;
    }
#endif
    printf("libllama unknown\n");
}

static uint32_t rd_u32(const unsigned char * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}

static void wr_u32(FILE * f, uint32_t v) {
    unsigned char b[4] = {
        (unsigned char) ( v        & 0xff),
        (unsigned char) ((v >>  8) & 0xff),
        (unsigned char) ((v >> 16) & 0xff),
        (unsigned char) ((v >> 24) & 0xff),
    };
    fwrite(b, 1, sizeof(b), f);
}

static void wr_i32(FILE * f, int32_t v) {
    wr_u32(f, (uint32_t) v);
}

static unsigned char * read_all(const char * path, size_t * out_len) {
    FILE * f = fopen(path, "rb");
    if (!f) {
        return NULL;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if (n < 0) {
        fclose(f);
        return NULL;
    }
    rewind(f);
    // One extra byte so the blob is always safe to treat as a buffer
    // even when the file is empty; the caller reads *out_len only.
    unsigned char * buf = (unsigned char *) malloc((size_t) n + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    if (n > 0 && fread(buf, 1, (size_t) n, f) != (size_t) n) {
        free(buf);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *out_len = (size_t) n;
    return buf;
}

static struct llama_model * load_model(const char * path, bool vocab_only) {
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.vocab_only   = vocab_only;

    struct llama_model * model = llama_model_load_from_file(path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load %s\n", path);
    }
    return model;
}

// ---------------------------------------------------------------- tokenize

static int cmd_tokenize(const char * model_path, const char * in_path, const char * out_path) {
    size_t blob_len = 0;
    unsigned char * blob = read_all(in_path, &blob_len);
    if (!blob) {
        fprintf(stderr, "cannot read case file %s\n", in_path);
        return 1;
    }
    if (blob_len < 8 || memcmp(blob, FXTK_MAGIC, 4) != 0) {
        fprintf(stderr, "%s is not an FXTK case file\n", in_path);
        free(blob);
        return 2;
    }

    const uint32_t n_cases = rd_u32(blob + 4);

    // Two passes: validate every length prefix before loading a model,
    // so a truncated case file fails in milliseconds and says so,
    // instead of failing after a multi-second load with a torn read.
    size_t * offsets = (size_t *) malloc(sizeof(size_t) * (size_t) (n_cases ? n_cases : 1));
    size_t * lengths = (size_t *) malloc(sizeof(size_t) * (size_t) (n_cases ? n_cases : 1));
    if (!offsets || !lengths) {
        fprintf(stderr, "out of memory for %u cases\n", n_cases);
        free(offsets);
        free(lengths);
        free(blob);
        return 1;
    }
    size_t at = 8;
    for (uint32_t i = 0; i < n_cases; i++) {
        if (at + 4 > blob_len) {
            fprintf(stderr, "case file truncated in the length of case %u\n", i);
            free(offsets);
            free(lengths);
            free(blob);
            return 2;
        }
        const uint32_t len = rd_u32(blob + at);
        at += 4;
        if (at + len > blob_len) {
            fprintf(stderr, "case file truncated in the body of case %u\n", i);
            free(offsets);
            free(lengths);
            free(blob);
            return 2;
        }
        offsets[i] = at;
        lengths[i] = len;
        at += len;
    }

    struct llama_model * model = load_model(model_path, true);
    if (!model) {
        free(offsets);
        free(lengths);
        free(blob);
        return EXIT_MODEL_UNSUPPORTED;
    }
    const struct llama_vocab * vocab = llama_model_get_vocab(model);

    FILE * f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "cannot open %s for writing\n", out_path);
        llama_model_free(model);
        free(offsets);
        free(lengths);
        free(blob);
        return 1;
    }

    uint32_t flags = 0;
    if (llama_vocab_get_add_bos(vocab)) flags |= 1u;
    if (llama_vocab_get_add_eos(vocab)) flags |= 2u;

    fwrite(FXTK_MAGIC, 1, 4, f);
    wr_u32(f, FXTK_VERSION);
    wr_u32(f, flags);
    wr_u32(f, (uint32_t) llama_vocab_n_tokens(vocab));
    wr_u32(f, n_cases);

    int rc = 0;
    for (uint32_t i = 0; i < n_cases && rc == 0; i++) {
        const char * text = (const char *) (blob + offsets[i]);
        const int32_t text_len = (int32_t) lengths[i];

        // One token per byte is the hard ceiling for any byte-fallback
        // vocabulary, so this never needs a retry; +4 covers the
        // specials llama.cpp may still emit for an empty case.
        int32_t cap = text_len + 4;
        llama_token * toks = (llama_token *) malloc(sizeof(llama_token) * (size_t) cap);
        if (!toks) {
            fprintf(stderr, "out of memory tokenizing case %u\n", i);
            rc = 1;
            break;
        }
        // add_special = false: BOS/EOS policy travels in `flags`.
        // parse_special: both, false first. See the format note above.
        for (int parse_special = 0; parse_special <= 1 && rc == 0; parse_special++) {
            const int32_t n = llama_tokenize(vocab, text, text_len, toks, cap, false, parse_special != 0);
            if (n < 0) {
                fprintf(stderr, "case %u needs %d tokens, buffer held %d\n", i, -n, cap);
                rc = 1;
                break;
            }
            wr_u32(f, (uint32_t) n);
            for (int32_t t = 0; t < n; t++) {
                wr_i32(f, (int32_t) toks[t]);
            }
        }
        free(toks);
    }

    fclose(f);
    llama_model_free(model);
    free(offsets);
    free(lengths);
    free(blob);

    if (rc == 0) {
        printf("tokenized %u cases\n", n_cases);
    }
    return rc;
}

// ------------------------------------------------------------------ logits

static int cmd_logits(const char * model_path, const char * out_path, int n_tokens, char ** tok_argv) {
    llama_token * tokens = (llama_token *) malloc(sizeof(llama_token) * (size_t) n_tokens);
    if (!tokens) {
        fprintf(stderr, "out of memory for %d tokens\n", n_tokens);
        return 1;
    }
    for (int i = 0; i < n_tokens; i++) {
        tokens[i] = (llama_token) atoi(tok_argv[i]);
    }

    struct llama_model * model = load_model(model_path, false);
    if (!model) {
        free(tokens);
        return EXIT_MODEL_UNSUPPORTED;
    }

    struct llama_context_params cparams = llama_context_default_params();
    // Context and batch must both cover the whole prompt: this decodes it
    // in ONE call, because splitting it would change the batch shape and
    // therefore which kernels llama.cpp picks — the thing under test.
    const char * steps_env = getenv("LLAMA_LOGITS_DECODE_STEPS");
    const int n_steps = steps_env ? atoi(steps_env) : 1;
    if (n_steps < 1) {
        fprintf(stderr, "LLAMA_LOGITS_DECODE_STEPS=%s is not a positive count\n", steps_env);
        free(tokens);
        llama_model_free(model);
        return 1;
    }
    cparams.n_ctx    = (uint32_t) (n_tokens + n_steps) + 8;
    cparams.n_batch  = (uint32_t) n_tokens;
    cparams.n_ubatch = (uint32_t) n_tokens;
    // LLAMA_LOGITS_FLASH_ATTN=0 keeps llama.cpp off its flash-attention
    // path. The default stays AUTO, which is what every published parity
    // number was measured under; the switch exists because llama.cpp
    // itself aborts under AUTO on a real PLM-1.8B file
    // (`ggml_set_rows: GGML_ASSERT(a->ne[0] == b->ne[0])` from
    // `build_attn` when the K head is 192 wide and the V head 128 --
    // measured 2026-09-12 against 1269cb1), and runs the same file with
    // it disabled, as `scripts/gptoss_reference_logits.cpp` always does.
    const char * fa = getenv("LLAMA_LOGITS_FLASH_ATTN");
    if (fa && strcmp(fa, "0") == 0) {
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    // FRINK_CTK is read by BOTH engines, so one variable decides the K/V
    // element type on each side of the comparison and the two cannot
    // disagree about it. Unset is f16, llama.cpp's default and frink's.
    // f32 is what `tools/oracle` runs: on the synthetic fixtures an f16
    // cache alone moves llama.cpp's own decode by KL 2.9e-2 (measured,
    // `phimoe_tiny` step 3, against itself with an f32 cache).
    const char * ctk = getenv("FRINK_CTK");
    if (ctk && strcmp(ctk, "f32") == 0) {
        cparams.type_k = GGML_TYPE_F32;
        cparams.type_v = GGML_TYPE_F32;
    } else if (ctk && strcmp(ctk, "f16") != 0) {
        fprintf(stderr, "FRINK_CTK=%s: the reference dumper takes f16 or f32\n", ctk);
        llama_model_free(model);
        free(tokens);
        return 1;
    }

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        free(tokens);
        return 1;
    }

    struct llama_batch batch = llama_batch_init((int32_t) n_tokens, 0, 1);
    for (size_t i = 0; i < (size_t) n_tokens; i++) {
        batch.token[i]     = tokens[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        // Only the last position's logits are wanted; asking for all of
        // them would allocate n_tokens*n_vocab floats for nothing.
        batch.logits[i]    = (i + 1 == (size_t) n_tokens);
    }
    batch.n_tokens = (int32_t) n_tokens;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        free(tokens);
        return 1;
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);

    const float * logits = llama_get_logits_ith(ctx, (int32_t) n_tokens - 1);
    if (!logits) {
        fprintf(stderr, "llama_get_logits_ith returned NULL\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        free(tokens);
        return 1;
    }

    FILE * f = fopen(out_path, "wb");
    if (!f) {
        fprintf(stderr, "cannot open %s for writing\n", out_path);
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        free(tokens);
        return 1;
    }
    fwrite(logits, sizeof(float), (size_t) n_vocab, f);
    for (int step = 1; step < n_steps; step++) {
        llama_token next = 0;
        for (int32_t v = 1; v < n_vocab; v++) {
            if (logits[v] > logits[next]) {
                next = v;
            }
        }
        struct llama_batch one = llama_batch_get_one(&next, 1);
        if (llama_decode(ctx, one) != 0) {
            fprintf(stderr, "llama_decode failed at decode step %d\n", step);
            fclose(f);
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            free(tokens);
            return 1;
        }
        logits = llama_get_logits_ith(ctx, -1);
        if (!logits) {
            fprintf(stderr, "llama_get_logits_ith returned NULL at decode step %d\n", step);
            fclose(f);
            llama_batch_free(batch);
            llama_free(ctx);
            llama_model_free(model);
            free(tokens);
            return 1;
        }
        fwrite(logits, sizeof(float), (size_t) n_vocab, f);
    }
    fclose(f);

    print_reference_identity();
    printf("n_vocab %d\n", n_vocab);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    free(tokens);
    return 0;
}

// ------------------------------------------------------------------- embed

// Pooled embedding for one text, through llama.cpp's own tokenizer
// (add_special = true, so a WPM vocab gets its [CLS]/[SEP]) and its own
// pooling type (whatever the GGUF's `{arch}.pooling_type` said).
static int cmd_embed(const char * model_path, const char * text) {
    struct llama_model * model = load_model(model_path, false);
    if (!model) {
        return EXIT_MODEL_UNSUPPORTED;
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    const int text_len = (int) strlen(text);
    int cap = text_len + 8;
    llama_token * toks = (llama_token *) malloc(sizeof(llama_token) * (size_t) cap);
    if (!toks) {
        fprintf(stderr, "out of memory\n");
        llama_model_free(model);
        return 1;
    }
    int32_t n_tokens = llama_tokenize(vocab, text, text_len, toks, cap, true, true);
    if (n_tokens < 0) {
        cap = -n_tokens;
        llama_token * grown = (llama_token *) realloc(toks, sizeof(llama_token) * (size_t) cap);
        if (!grown) {
            fprintf(stderr, "out of memory\n");
            free(toks);
            llama_model_free(model);
            return 1;
        }
        toks = grown;
        n_tokens = llama_tokenize(vocab, text, text_len, toks, cap, true, true);
    }
    if (n_tokens <= 0) {
        fprintf(stderr, "tokenization produced %d tokens\n", (int) n_tokens);
        free(toks);
        llama_model_free(model);
        return 1;
    }

    struct llama_context_params cparams = llama_context_default_params();
    cparams.embeddings = true;
    // A non-causal model cannot be split across micro-batches: every
    // token attends to every other one, so n_ubatch must cover the whole
    // sequence. Same reason the logits mode decodes in one call.
    cparams.n_ctx    = (uint32_t) n_tokens + 8;
    cparams.n_batch  = (uint32_t) n_tokens + 8;
    cparams.n_ubatch = (uint32_t) n_tokens + 8;

    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to create context\n");
        free(toks);
        llama_model_free(model);
        return 1;
    }

    struct llama_batch batch = llama_batch_init(n_tokens, 0, 1);
    for (int32_t i = 0; i < n_tokens; i++) {
        batch.token[i]     = toks[i];
        batch.pos[i]       = (llama_pos) i;
        batch.n_seq_id[i]  = 1;
        batch.seq_id[i][0] = 0;
        // Pooled output is produced only for positions flagged for
        // output; the upstream embedding example flags every one.
        batch.logits[i]    = 1;
    }
    batch.n_tokens = n_tokens;

    int rc = 0;
    if (llama_decode(ctx, batch) < 0) {
        fprintf(stderr, "llama_decode failed\n");
        rc = 1;
    } else {
        const enum llama_pooling_type pooling = llama_pooling_type(ctx);
        const float * embd = (pooling == LLAMA_POOLING_TYPE_NONE)
            ? llama_get_embeddings_ith(ctx, 0)
            : llama_get_embeddings_seq(ctx, 0);
        if (!embd) {
            fprintf(stderr, "no embeddings returned (pooling_type %d)\n", (int) pooling);
            rc = 1;
        } else {
            const int32_t n_embd = llama_model_n_embd(model);
            fprintf(stderr, "pooling_type %d, n_tokens %d, ids:", (int) pooling, (int) n_tokens);
            for (int32_t i = 0; i < n_tokens; i++) {
                fprintf(stderr, " %d", (int) toks[i]);
            }
            fprintf(stderr, "\n");
            for (int32_t i = 0; i < n_embd; i++) {
                printf("%.9g\n", (double) embd[i]);
            }
        }
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    free(toks);
    return rc;
}

int main(int argc, char ** argv) {
    if (argc >= 2 && strcmp(argv[1], "--tokenize") == 0) {
        if (argc != 5) {
            fprintf(stderr, "usage: %s --tokenize <model.gguf> <cases.bin> <out.bin>\n", argv[0]);
            return 2;
        }
        llama_backend_init();
        const int rc = cmd_tokenize(argv[2], argv[3], argv[4]);
        llama_backend_free();
        return rc;
    }

    if (argc >= 2 && strcmp(argv[1], "--embed") == 0) {
        if (argc != 4) {
            fprintf(stderr, "usage: %s --embed <model.gguf> <text>\n", argv[0]);
            return 2;
        }
        llama_backend_init();
        const int rc = cmd_embed(argv[2], argv[3]);
        llama_backend_free();
        return rc;
    }

    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <out.bin> <tok0> [tok1 ...]\n", argv[0]);
        fprintf(stderr, "       %s --tokenize <model.gguf> <cases.bin> <out.bin>\n", argv[0]);
        fprintf(stderr, "       %s --embed <model.gguf> <text>\n", argv[0]);
        return 2;
    }

    llama_backend_init();
    const int rc = cmd_logits(argv[1], argv[2], argc - 3, argv + 3);
    llama_backend_free();
    return rc;
}
