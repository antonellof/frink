// Calibration driver for tc_shim.h: the tensor-core GEMM body
// (`frink_cuda::mul_mm_tc`, Q8_0), which passed against its reference on
// real hardware, through the PTX stand-ins. Its output must equal the
// exact f16 product summed in k order (the example's `gemm.want`) bit
// for bit, or the stand-ins are not the instructions and the attention
// check that follows proves nothing. The kind's body is textually
// included ahead of this file, with -DFX_KERNEL naming it (see run.sh).
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

static std::vector<unsigned char> readb(const std::string& p) {
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
        fprintf(stderr, "usage: %s <base> <n_rows> <n_cols> <batch> <row_bytes>\n", argv[0]);
        return 2;
    }
    const std::string base = argv[1];
    const int n_rows = atoi(argv[2]), n_cols = atoi(argv[3]);
    const int batch = atoi(argv[4]), row_bytes = atoi(argv[5]);
    auto w = readb(base + ".w");
    auto xb = readb(base + ".x");
    auto wantb = readb(base + ".want");
    const unsigned short* x = (const unsigned short*)xb.data();
    const float* want = (const float*)wantb.data();
    std::vector<float> dst((size_t)batch * n_rows, std::numeric_limits<float>::quiet_NaN());
    if (wantb.size() != dst.size() * 4) {
        fprintf(stderr, "%s: expected-output length mismatch\n", base.c_str());
        return 2;
    }

    const unsigned threads = FX_TTHREADS;
    const unsigned gx = (batch + FX_TBN - 1) / FX_TBN, gy = (n_rows + FX_TBM - 1) / FX_TBM;
    blockDim = {threads, 1, 1};
    gridDim = {gx, gy, 1};
    for (unsigned by = 0; by < gy; by++) {
        for (unsigned bx = 0; bx < gx; bx++) {
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
                    FX_KERNEL(w.data(), x, dst.data(), n_rows, n_cols, batch, row_bytes);
                });
            }
            for (auto& th : ts) th.join();
        }
    }

    size_t mismatch = 0;
    for (size_t i = 0; i < dst.size(); i++) {
        if (!(dst[i] == want[i])) {
            mismatch++;
            if (mismatch <= 3)
                fprintf(stderr, "gemm[%zu]: kernel=%.9g exact=%.9g\n", i, dst[i], want[i]);
        }
    }
    printf("calibration: the tensor-core GEMM body, %dx%d over %d tokens, mismatches=%zu\n",
           n_rows, n_cols, batch, mismatch);
    return mismatch == 0 ? 0 : 1;
}
