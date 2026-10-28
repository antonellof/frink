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

int main(int argc, char** argv) {
    if (argc != 10) {
        fprintf(stderr,
                "usage: %s <base> <n_heads> <n_kv> <head_dim> <lo> <hi> <n_chunks> "
                "<scale_bits> <softcap_bits>\n",
                argv[0]);
        return 2;
    }
    const std::string base = argv[1];
    const int n_heads = atoi(argv[2]), n_kv = atoi(argv[3]), hd = atoi(argv[4]);
    const int lo = atoi(argv[5]), hi = atoi(argv[6]), n_chunks = atoi(argv[7]);
    const float scale = from_bits(argv[8]), softcap = from_bits(argv[9]);

    auto q = readf(base + ".q");
    auto k = readf(base + ".k");
    auto v = readf(base + ".v");
    auto want = readf(base + ".want");
    std::vector<float> part((size_t)n_heads * n_chunks * (hd + 2), 0.0f);
    std::vector<float> out((size_t)n_heads * hd, 0.0f);

    run_grid((unsigned)n_heads, (unsigned)n_chunks, 128, [&]() {
        decode_attn_partial_f32(q.data(), k.data(), v.data(), part.data(), n_heads, n_kv, hd, lo,
                                hi, n_chunks, scale, softcap);
    });
    run_grid((unsigned)n_heads, 1, (unsigned)hd, [&]() {
        decode_attn_combine_f32(part.data(), out.data(), n_heads, hd, n_chunks);
    });

    if (want.size() != out.size()) {
        fprintf(stderr, "%s: expected-output length mismatch\n", base.c_str());
        return 2;
    }
    size_t mismatch = 0;
    for (size_t i = 0; i < out.size(); i++) {
        const bool gn = std::isnan(out[i]), wn = std::isnan(want[i]);
        const bool same = (gn || wn) ? (gn && wn) : out[i] == want[i];
        if (!same) {
            mismatch++;
            if (mismatch <= 3)
                fprintf(stderr, "%s[%zu]: kernel=%.9g twin=%.9g\n", base.c_str(), i, out[i],
                        want[i]);
        }
    }
    printf("%-8s heads=%d hd=%3d keys=%d..%d chunks=%d mismatches=%zu\n", base.c_str(), n_heads,
           hd, lo, hi, n_chunks, mismatch);
    return mismatch == 0 ? 0 : 1;
}
