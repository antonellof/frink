// Driver for tc_shim.h, for the tensor-core prefill attention
// (`frink_cuda::prefill_attn_tc`). One head width's body is textually
// included ahead of this file, with -DFX_KERNEL naming its entry point
// (see run.sh).
//
// Compares bit-exactly against the Rust twin: with FP contraction off
// the body and the twin perform the identical sequence of f32
// operations. Every output element starts as a NaN, so an element the
// kernel never writes is a mismatch too.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
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
                wb.emplace_back(new Barrier(32));
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
    if (argc != 13) {
        fprintf(stderr,
                "usage: %s <base> <n_q> <n_heads> <n_kv> <start_pos> <window> <qscale_bits> "
                "<softcap_bits> <group> <grid_x> <grid_y> <smem_bytes>\n",
                argv[0]);
        return 2;
    }
    const std::string base = argv[1];
    const int n_q = atoi(argv[2]), n_heads = atoi(argv[3]), n_kv = atoi(argv[4]);
    const int start_pos = atoi(argv[5]), window = atoi(argv[6]);
    const float qscale = from_bits(argv[7]), softcap = from_bits(argv[8]);
    const int group = atoi(argv[9]);
    const unsigned gx = (unsigned)atoi(argv[10]), gy = (unsigned)atoi(argv[11]);
    const size_t smem = (size_t)atoi(argv[12]);
    if (smem > sizeof(g_tc_dyn_smem)) {
        fprintf(stderr, "%s: %zu shared bytes, the stand-in holds %zu\n", base.c_str(), smem,
                sizeof(g_tc_dyn_smem));
        return 2;
    }

    auto q = readf(base + ".q");
    auto k = readf(base + ".k");
    auto v = readf(base + ".v");
    auto want = readf(base + ".want");
    auto sinks = readf(base + ".sinks");
    const float* sinks_p = sinks.empty() ? nullptr : sinks.data();
    std::vector<float> out(want.size(), std::numeric_limits<float>::quiet_NaN());

    run_grid(gx, gy, FX_THREADS, [&]() {
        FX_KERNEL(q.data(), k.data(), v.data(), out.data(), sinks_p, n_q, n_heads | (n_kv << 16),
                  start_pos, window, qscale, softcap, group);
    });

    size_t mismatch = 0;
    for (size_t i = 0; i < out.size(); i++) {
        if (!(out[i] == want[i])) {
            mismatch++;
            if (mismatch <= 3)
                fprintf(stderr, "%s[%zu]: kernel=%.9g twin=%.9g\n", base.c_str(), i, out[i],
                        want[i]);
        }
    }
    printf("%-10s n_q=%3d heads=%d/%d start=%3d window=%3d group=%d sinks=%s mismatches=%zu\n",
           base.substr(base.find_last_of('/') + 1).c_str(), n_q, n_heads, n_kv, start_pos, window,
           group, sinks_p ? "yes" : "no", mismatch);
    return mismatch == 0 ? 0 : 1;
}
