// Driver for ../mul_mm_host_check/shim.h, for the one-row attention
// kernels (`frink_cuda::decode_attn`). The kernel text is textually
// included ahead of this file (see run.sh).
//
// Compares bit-exactly against the Rust twin: with FP contraction off
// the kernels and the twin perform the identical sequence of fp32
// operations, the shuffle butterfly and both merges included.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

static std::vector<float> readf(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        fprintf(stderr, "missing %s\n", p.c_str());
        exit(2);
    }
    std::vector<unsigned char> b((std::istreambuf_iterator<char>(f)),
                                 std::istreambuf_iterator<char>());
    std::vector<float> out(b.size() / 4);
    memcpy(out.data(), b.data(), out.size() * 4);
    return out;
}

static float from_bits(const char* s) {
    uint32_t u = (uint32_t)strtoul(s, nullptr, 10);
    float f;
    memcpy(&f, &u, 4);
    return f;
}

// Runs `body` as a grid of `grid_x * grid_y` blocks of `threads`, one
// block at a time, with a block barrier and one barrier per warp.
template <typename F>
static void run_grid(unsigned grid_x, unsigned grid_y, unsigned threads, F body) {
    blockDim = {threads, 1, 1};
    gridDim = {grid_x, grid_y, 1};
    const unsigned warps = (threads + 31) / 32;
    for (unsigned by = 0; by < grid_y; by++) {
        for (unsigned bx = 0; bx < grid_x; bx++) {
            Barrier bar(threads);
            g_bar = &bar;
            std::vector<std::unique_ptr<Barrier>> wb;
            std::vector<Barrier*> wbp;
            for (unsigned i = 0; i < warps; i++) {
                const unsigned in_warp = (i + 1) * 32 <= threads ? 32 : threads - i * 32;
                wb.emplace_back(new Barrier(in_warp));
                wbp.push_back(wb.back().get());
            }
            g_warp_bar = wbp.data();
            std::vector<std::thread> ts;
            for (unsigned t = 0; t < threads; t++) {
                ts.emplace_back([&, t]() {
                    blockIdx = {bx, by, 0};
                    threadIdx = {t, 0, 0};
                    body();
                });
            }
            for (auto& th : ts) th.join();
        }
    }
}

// Runs the pair at `n_slots` chunks of grid, the position by value or,
// when `pos` is not null, from the word it points at.
static std::vector<float> run_pair(const std::vector<float>& q, const std::vector<float>& k,
                                   const std::vector<float>& v, const float* sinks, int n_heads,
                                   int n_kv, int hd, int seq_arg, int window, int n_slots,
                                   float scale, float softcap, const int* pos) {
    // NaN, so a slot a kernel should have written and did not shows.
    std::vector<float> part((size_t)n_heads * n_slots * (hd + 2), NAN);
    std::vector<float> out((size_t)n_heads * hd, 0.0f);
    run_grid((unsigned)n_heads, (unsigned)n_slots, 128, [&]() {
        decode_attn_partial_f32(q.data(), k.data(), v.data(), part.data(), n_heads | (n_kv << 16),
                                hd, seq_arg, window, n_slots, scale, softcap, pos);
    });
    run_grid((unsigned)n_heads, 1, (unsigned)hd, [&]() {
        decode_attn_combine_f32(part.data(), out.data(), n_heads, hd, n_slots, sinks, seq_arg,
                                window, pos);
    });
    return out;
}

static size_t mismatches(const std::string& tag, const std::vector<float>& out,
                         const std::vector<float>& want) {
    if (want.size() != out.size()) {
        fprintf(stderr, "%s: expected-output length mismatch\n", tag.c_str());
        exit(2);
    }
    size_t mismatch = 0;
    for (size_t i = 0; i < out.size(); i++) {
        const bool gn = std::isnan(out[i]), wn = std::isnan(want[i]);
        const bool same = (gn || wn) ? (gn && wn) : out[i] == want[i];
        if (!same) {
            mismatch++;
            if (mismatch <= 3)
                fprintf(stderr, "%s[%zu]: kernel=%.9g twin=%.9g\n", tag.c_str(), i, out[i],
                        want[i]);
        }
    }
    return mismatch;
}

int main(int argc, char** argv) {
    if (argc != 11) {
        fprintf(stderr,
                "usage: %s <base> <n_heads> <n_kv> <head_dim> <seq_len> <window> <n_chunks> "
                "<replay_slots> <scale_bits> <softcap_bits>\n",
                argv[0]);
        return 2;
    }
    const std::string base = argv[1];
    const int n_heads = atoi(argv[2]), n_kv = atoi(argv[3]), hd = atoi(argv[4]);
    const int seq_len = atoi(argv[5]), window = atoi(argv[6]);
    const int n_chunks = atoi(argv[7]), slots = atoi(argv[8]);
    const float scale = from_bits(argv[9]), softcap = from_bits(argv[10]);

    auto q = readf(base + ".q");
    auto k = readf(base + ".k");
    auto v = readf(base + ".v");
    auto want = readf(base + ".want");
    auto sinks = readf(base + ".sinks");
    const float* sinks_p = sinks.empty() ? nullptr : sinks.data();

    // Enqueued as it runs: the position by value, exactly its chunks.
    auto eager = run_pair(q, k, v, sinks_p, n_heads, n_kv, hd, seq_len, window, n_chunks, scale,
                          softcap, nullptr);
    // Replayed: the position from a device word, the grid as wide as the
    // capacity, and a by-value position that must not be read.
    const int pos = seq_len - 1;
    auto replay = run_pair(q, k, v, sinks_p, n_heads, n_kv, hd, -12345, window, slots, scale,
                           softcap, &pos);

    const size_t me = mismatches(base + " eager", eager, want);
    const size_t mr = mismatches(base + " replay", replay, want);
    printf("%-8s heads=%d hd=%3d keys=..%d window=%d chunks=%d slots=%d sinks=%s "
           "mismatches=%zu/%zu\n",
           base.c_str(), n_heads, hd, seq_len, window, n_chunks, slots, sinks_p ? "yes" : "no",
           me, mr);
    return me == 0 && mr == 0 ? 0 : 1;
}
