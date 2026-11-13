// Driver for ../mul_mm_host_check/shim.h, for the routed-expert kernels
// (`mul_mv_id`, `mul_mm_id`). Built once per (kind, kernel), with
// -DFX_KERNEL set to the entry point, -DFX_MODE_MV or -DFX_MODE_MM, and
// that kernel's generated translation unit textually included ahead of
// this file (see run.sh). One block at a time, one std::thread per CUDA
// thread, as the other host checks do. Compared BIT for bit (NaN on
// both sides is agreement).
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

template <typename F>
static void run_block(Dim3 b, unsigned threads, F body) {
    Barrier bar(threads);
    g_bar = &bar;
    std::vector<std::thread> ts;
    for (unsigned t = 0; t < threads; t++) {
        ts.emplace_back([&, t]() {
            blockIdx = b;
            threadIdx = {t, 0, 0};
            body();
        });
    }
    for (auto& th : ts) th.join();
}

int main(int argc, char** argv) {
    // mv: <base> <rows> <cols> <row_bytes> <n_blocks> <n_slots> <x_div> <expert_stride>
    // mm: <base> <rows> <cols> <row_bytes> <n_out> <n_active> <max_count> <expert_stride>
    if (argc != 9) {
        fprintf(stderr, "usage: %s <base> <7 shape arguments>\n", argv[0]);
        return 2;
    }
    const std::string base = argv[1];
    const int rows = atoi(argv[2]), cols = atoi(argv[3]), row_bytes = atoi(argv[4]);
    const int a5 = atoi(argv[5]), a6 = atoi(argv[6]), a7 = atoi(argv[7]);
    const unsigned long long stride = strtoull(argv[8], nullptr, 10);

    auto w = readf(base + ".w");
    auto xb = readf(base + ".x");
    auto wantb = readf(base + ".want");
    const float* x = (const float*)xb.data();
    const float* want = (const float*)wantb.data();
    const size_t n = wantb.size() / 4;
    std::vector<float> out(n, NAN);

#if defined(FX_MODE_MV)
    auto idb = readf(base + ".ids");
    const int* ids = (const int*)idb.data();
    const int n_blocks = a5, n_slots = a6, x_div = a7;
    blockDim = {256, 1, 1};
    gridDim = {(unsigned)rows, (unsigned)n_slots, 1};
    for (int s = 0; s < n_slots; s++)
        for (int r = 0; r < rows; r++)
            run_block({(unsigned)r, (unsigned)s, 0}, 256, [&]() {
                FX_KERNEL(w.data(), x, out.data(), ids, rows, row_bytes, n_blocks, n_slots, x_div,
                          cols, stride);
            });
#elif defined(FX_MODE_MM)
    auto mb = readf(base + ".mb"), md = readf(base + ".md"), eb = readf(base + ".experts");
    const int n_active = a6, max_count = a7;
    const unsigned threads = FX_THREADS;
    const unsigned gx = (max_count + FX_BN - 1) / FX_BN, gy = (rows + FX_BM - 1) / FX_BM;
    blockDim = {threads, 1, 1};
    gridDim = {gx, gy, (unsigned)n_active};
    for (int z = 0; z < n_active; z++)
        for (unsigned by = 0; by < gy; by++)
            for (unsigned bx = 0; bx < gx; bx++)
                run_block({bx, by, (unsigned)z}, threads, [&]() {
                    FX_KERNEL(w.data(), x, out.data(), rows, cols, max_count, row_bytes,
                              (const int*)mb.data(), (const int*)md.data(),
                              (const int*)eb.data(), stride);
                });
    (void)a5;
#else
#error "define FX_MODE_MV or FX_MODE_MM"
#endif

    size_t mismatch = 0, nan_both = 0;
    for (size_t i = 0; i < n; i++) {
        unsigned a, b;
        memcpy(&a, &out[i], 4);
        memcpy(&b, &want[i], 4);
        if (std::isnan(out[i]) && std::isnan(want[i])) {
            nan_both++;
            continue;
        }
        if (a != b) {
            mismatch++;
            if (mismatch <= 3)
                fprintf(stderr, "%s[%zu]: kernel=%.9g twin=%.9g\n", base.c_str(), i, out[i],
                        want[i]);
        }
    }
    printf("%-36s n=%6zu mismatches=%zu (nan-agreements=%zu)\n", base.c_str(), n, mismatch,
           nan_both);
    return mismatch == 0 ? 0 : 1;
}
