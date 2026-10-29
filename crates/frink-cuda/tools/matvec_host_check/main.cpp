// Driver for ../mul_mm_host_check/shim.h, for the coalesced matvec
// kernels. Built once per (kind, width), with -DFX_KERNEL set to the
// entry point and that width's generated translation unit textually
// included ahead of this file (see run.sh).
//
// Compares bit-exactly against the Rust twin: with FP contraction off
// the kernel and the twin perform the identical sequence of fp32
// operations, including the shuffle tree, so anything short of
// equality is a real divergence. NaN on both sides is agreement.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

static std::vector<unsigned char> readf(const std::string& p) {
    std::ifstream f(p, std::ios::binary);
    if (!f) {
        fprintf(stderr, "missing %s\n", p.c_str());
        exit(2);
    }
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(f)),
                                      std::istreambuf_iterator<char>());
}

// The launch geometry `gpu.rs::matvec_launch_plan` uses: one row per
// warp, eight warps per block. Restated here because this tool cannot
// call Rust; `run.sh` passes the value the example read from the crate.
int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <base> <rows> <row_bytes> <n_blocks> <warps>\n", argv[0]);
        return 2;
    }
    const std::string base = argv[1];
    const int rows = atoi(argv[2]), row_bytes = atoi(argv[3]);
    const int n_blocks = atoi(argv[4]);
    const unsigned warps = (unsigned)atoi(argv[5]);

    auto w = readf(base + ".w");
    auto xb = readf(base + ".x");
    auto wantb = readf(base + ".want");
    const float* x = (const float*)xb.data();
    const float* want = (const float*)wantb.data();
    std::vector<float> out(rows, 0.0f);

    const unsigned threads = warps * 32;
    const unsigned grid = (rows + warps - 1) / warps;
    blockDim = {threads, 1, 1};
    gridDim = {grid, 1, 1};

    for (unsigned bx = 0; bx < grid; bx++) {
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
                blockIdx = {bx, 0, 0};
                threadIdx = {t, 0, 0};
                FX_KERNEL(w.data(), x, out.data(), rows, row_bytes, n_blocks);
            });
        }
        for (auto& th : ts) th.join();
    }

    if (wantb.size() != size_t(rows) * 4) {
        fprintf(stderr, "%s: expected-output length mismatch\n", base.c_str());
        return 2;
    }
    size_t nan_both = 0, mismatch = 0;
    for (int i = 0; i < rows; i++) {
        const bool gn = std::isnan(out[i]), wn = std::isnan(want[i]);
        const bool same = (gn || wn) ? (gn && wn) : out[i] == want[i];
        if (gn && wn) nan_both++;
        if (!same) {
            mismatch++;
            if (mismatch <= 3)
                fprintf(stderr, "%s[%d]: kernel=%.9g twin=%.9g\n", base.c_str(), i, out[i],
                        want[i]);
        }
    }
    printf("%-36s rows=%3d mismatches=%zu (nan-agreements=%zu)\n", base.c_str(), rows, mismatch,
           nan_both);
    return mismatch == 0 ? 0 : 1;
}
