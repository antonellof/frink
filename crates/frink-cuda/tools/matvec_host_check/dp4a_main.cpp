// Driver for ../mul_mm_host_check/shim.h, for the q8_1 quantizer and the
// dp4a matvecs (`matvec_kinds/dp4a.rs`). Built once per translation
// unit with -DFX_KERNEL set to the entry point, the unit textually
// included ahead of this file (see run.sh), and -DFQ_QUANT_DRIVER for
// the quantizer.
//
// Bit-exact against the Rust twins (`frink_cuda::dp4a_twin`): with FP
// contraction off the kernel and the twin perform the same fp32
// operations in the same order, the integer dots are exact, and the
// quantizer's bytes are compared whole. NaN on both sides is agreement.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// One threadblock of `threads` host threads at (bx, by), with the
// block-wide barrier `__syncthreads` meets at and one barrier per warp
// for the shuffles.
template <class F>
static void run_block(unsigned threads, unsigned bx, unsigned by, F body) {
    Barrier bar(threads);
    g_bar = &bar;
    std::vector<std::unique_ptr<Barrier>> wb;
    std::vector<Barrier*> wbp;
    for (unsigned i = 0; i < threads / 32; i++) {
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

#ifdef FQ_QUANT_DRIVER
// usage: <base> <n> <threads>
int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s <base> <n> <threads>\n", argv[0]);
        return 2;
    }
    const std::string base = argv[1];
    const int n = atoi(argv[2]);
    const unsigned threads = (unsigned)atoi(argv[3]);
    auto xb = readf(base + ".x");
    auto want = readf(base + ".want");
    const float* x = (const float*)xb.data();
    std::vector<unsigned char> y(want.size(), 0xA5);
    const unsigned grid = ((unsigned)n + threads - 1) / threads;
    blockDim = {threads, 1, 1};
    gridDim = {grid, 1, 1};
    for (unsigned bx = 0; bx < grid; bx++) {
        run_block(threads, bx, 0, [&]() { FX_KERNEL(x, y.data(), n); });
    }
    size_t mismatch = 0;
    for (size_t i = 0; i < want.size(); i++) {
        if (y[i] != want[i]) {
            if (++mismatch <= 3)
                fprintf(stderr, "%s byte %zu: kernel=%02x twin=%02x\n", base.c_str(), i, y[i],
                        want[i]);
        }
    }
    printf("%-44s n=%5d mismatched bytes=%zu\n", base.c_str(), n, mismatch);
    return mismatch == 0 ? 0 : 1;
}
#else
// usage: <base> <rows> <row_bytes> <n_blocks> <threads> <rows_per_block> <slots> <y_stride>
int main(int argc, char** argv) {
    if (argc != 9) {
        fprintf(stderr,
                "usage: %s <base> <rows> <row_bytes> <n_blocks> <threads> <rows_per_block> "
                "<slots> <y_stride>\n",
                argv[0]);
        return 2;
    }
    const std::string base = argv[1];
    const int rows = atoi(argv[2]), row_bytes = atoi(argv[3]), n_blocks = atoi(argv[4]);
    const unsigned threads = (unsigned)atoi(argv[5]);
    const int rpb = atoi(argv[6]), slots = atoi(argv[7]), y_stride = atoi(argv[8]);
    auto w = readf(base + ".w");
    auto y = readf(base + ".y");
    auto wantb = readf(base + ".want");
    const float* want = (const float*)wantb.data();
    if (wantb.size() != size_t(rows) * size_t(slots) * 4 ||
        w.size() != size_t(rows) * size_t(row_bytes) * size_t(slots)) {
        fprintf(stderr, "%s: fixture length mismatch\n", base.c_str());
        return 2;
    }
    std::vector<float> out(size_t(rows) * size_t(slots), 0.0f);
    const unsigned grid = (unsigned)((rows + rpb - 1) / rpb);
    blockDim = {threads, 1, 1};
    gridDim = {grid, (unsigned)slots, 1};
#if FX_MULTI
    // The routed form reads slot `s`'s weights through a pointer table.
    std::vector<unsigned long long> table;
    for (int s = 0; s < slots; s++)
        table.push_back((unsigned long long)(w.data() + size_t(s) * rows * row_bytes));
#else
    if (slots != 1 || y_stride != 0) {
        fprintf(stderr, "%s: a single-matrix kernel takes one slot\n", base.c_str());
        return 2;
    }
#endif
    for (unsigned by = 0; by < (unsigned)slots; by++) {
        for (unsigned bx = 0; bx < grid; bx++) {
            run_block(threads, bx, by, [&]() {
#if FX_MULTI
                FX_KERNEL(table.data(), y.data(), out.data(), rows, row_bytes, n_blocks, y_stride);
#else
                FX_KERNEL(w.data(), y.data(), out.data(), rows, row_bytes, n_blocks);
#endif
            });
        }
    }
    size_t nan_both = 0, mismatch = 0;
    for (size_t i = 0; i < out.size(); i++) {
        const bool gn = std::isnan(out[i]), wn = std::isnan(want[i]);
        const bool same = (gn || wn) ? (gn && wn) : out[i] == want[i];
        if (gn && wn) nan_both++;
        if (!same && ++mismatch <= 3)
            fprintf(stderr, "%s[%zu]: kernel=%.9g twin=%.9g\n", base.c_str(), i, out[i], want[i]);
    }
    printf("%-44s rows=%3d x%d mismatches=%zu (nan-agreements=%zu)\n", base.c_str(), rows, slots,
           mismatch, nan_both);
    return mismatch == 0 ? 0 : 1;
}
#endif
