// Driver for ../mul_mm_host_check/shim.h, for the get_rows kernels.
// Built once per quant kind, with -DFX_KERNEL set to the entry point
// and the kind's generated translation unit textually included ahead
// of this file (see run.sh).
//
// The gather has no barrier and no shared memory, so every CUDA thread
// runs to completion on this one host thread in turn; the grid is the
// one `get_rows::grid_dims` computes, restated here from the manifest.
// Compared BIT for bit (NaN on both sides is agreement).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
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

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "usage: %s <base> <n_ids> <n_cols> <row_bytes> <threads>\n", argv[0]);
        return 2;
    }
    const std::string base = argv[1];
    const int n_ids = atoi(argv[2]), n_cols = atoi(argv[3]), row_bytes = atoi(argv[4]);
    const unsigned threads = (unsigned)atoi(argv[5]);

    auto w = readf(base + ".w");
    auto idb = readf(base + ".ids");
    auto wantb = readf(base + ".want");
    const unsigned* ids = (const unsigned*)idb.data();
    const float* want = (const float*)wantb.data();
    const size_t n = size_t(n_ids) * n_cols;
    // Poisoned, so an element the kernel never writes is a mismatch.
    std::vector<float> dst(n, NAN);
    if (wantb.size() != n * 4 || idb.size() != size_t(n_ids) * 4) {
        fprintf(stderr, "%s: fixture length mismatch\n", base.c_str());
        return 2;
    }

    const unsigned gx = (unsigned)((n_cols / 16 + threads - 1) / threads);
    blockDim = {threads, 1, 1};
    gridDim = {gx, (unsigned)n_ids, 1};
    for (unsigned by = 0; by < (unsigned)n_ids; by++) {
        for (unsigned bx = 0; bx < gx; bx++) {
            for (unsigned t = 0; t < threads; t++) {
                blockIdx = {bx, by, 0};
                threadIdx = {t, 0, 0};
                FX_KERNEL(w.data(), ids, dst.data(), n_ids, n_cols, row_bytes);
            }
        }
    }

    size_t mismatch = 0, nan_both = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned a, b;
        memcpy(&a, &dst[i], 4);
        memcpy(&b, &want[i], 4);
        const bool gn = std::isnan(dst[i]), wn = std::isnan(want[i]);
        if (gn && wn) {
            nan_both++;
            continue;
        }
        if (a != b) {
            mismatch++;
            if (mismatch <= 3)
                fprintf(stderr, "%s[%zu]: kernel=%.9g twin=%.9g\n", base.c_str(), i, dst[i],
                        want[i]);
        }
    }
    printf("%-36s n=%6zu mismatches=%zu (nan-agreements=%zu)\n", base.c_str(), n, mismatch,
           nan_both);
    return mismatch == 0 ? 0 : 1;
}
