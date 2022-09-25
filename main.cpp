#include <benchmark/benchmark.h>

#include <vector>

#include <cassert>
#include <cstdint>


constexpr const std::size_t SIZE = 2 << 20;
static const std::vector<uint8_t> data = []() {
    std::vector<uint8_t> out(SIZE);
    for (std::size_t i=0; i < SIZE; ++i)
        out[i] = i;
    return out;
}();


static void f1(benchmark::State& state) {
    std::vector<uint8_t> out(SIZE);
    for (auto _: state)
        for (std::size_t i=0; i < SIZE; ++i) {
            out[i] = 255.0f * (data[i] / 128.0f);
        }
    benchmark::DoNotOptimize(out);
}
BENCHMARK(f1);


static void f2(benchmark::State& state) {
    std::vector<uint8_t> out(SIZE);

    const uint8_t* ind = data.data();
    uint8_t* outd = out.data();

    for (auto _: state)
        for (std::size_t i=0; i < SIZE; ++i) {
            outd[i] = static_cast<uint8_t>(255.0f * (ind[i] / 128.0f));
        }
    benchmark::DoNotOptimize(out);
}
BENCHMARK(f2);


#include <emmintrin.h>
#include <smmintrin.h>

static void f3(benchmark::State& state) {
    assert(SIZE % 4 == 0);

    std::vector<uint8_t> out(SIZE);

    const auto* ind = reinterpret_cast<const uint32_t*>(data.data());
    auto* outd = reinterpret_cast<uint32_t*>(out.data());

    for (auto _: state)
        for (std::size_t i=0; i < SIZE / 4; ++i) {
            const auto v = ind[i];
            __m128i u32x4 = _mm_cvtepu8_epi32(__m128i{v});
            __m128 fx4 = _mm_cvtepi32_ps(u32x4);
            fx4 = 255.0f * (fx4 / 128.0f);
            u32x4 = _mm_cvttps_epi32(fx4);
            const auto v2 =
                    (u32x4[1] >> 32)  << 24 |
                    (u32x4[1] & 0xff) << 16 |
                    (u32x4[0] >> 32)  <<  8 |
                    (u32x4[0] & 0xff);
            outd[i] = v2;
        }
    benchmark::DoNotOptimize(out);
}
BENCHMARK(f3);


static void f4(benchmark::State& state) {
    assert(SIZE % 4 == 0);

    std::vector<uint8_t> out(SIZE);

    const auto* ind = reinterpret_cast<const uint32_t*>(data.data());
    auto* outd = reinterpret_cast<uint32_t*>(out.data());

    for (auto _: state)
        for (std::size_t i=0; i < SIZE / 4; ++i) {
            const auto v = ind[i];
            __m128i u32x4 = _mm_cvtepu8_epi32(__m128i{v});
            __m128 fx4 = _mm_cvtepi32_ps(u32x4);
            fx4 = 255.0f * (fx4 / 128.0f);
            u32x4 = _mm_cvttps_epi32(fx4);
            const auto v2 =
                    _mm_extract_epi8(u32x4, 0) << 24 |
                    _mm_extract_epi8(u32x4, 4) << 16 |
                    _mm_extract_epi8(u32x4, 8) <<  8 |
                    _mm_extract_epi8(u32x4, 12);
            outd[i] = v2;
        }
    benchmark::DoNotOptimize(out);
}
BENCHMARK(f4);


static void f5(benchmark::State& state) {
    assert(SIZE % 4 == 0);

    std::vector<uint8_t> out(SIZE);

    const auto* ind = reinterpret_cast<const uint32_t*>(data.data());
    auto* outd = reinterpret_cast<uint32_t*>(out.data());

    for (auto _: state)
        for (std::size_t i=0; i < SIZE / 4; ++i) {
            const auto v = ind[i];
            __m128i u32x4 = _mm_cvtepu8_epi32(__m128i{v});
            __m128 fx4 = _mm_cvtepi32_ps(u32x4);
            fx4 = 255.0f * (fx4 / 128.0f);
            u32x4 = _mm_cvttps_epi32(fx4);
            u32x4 = _mm_packus_epi32(u32x4, u32x4);
            u32x4 = _mm_packus_epi16(u32x4, u32x4);

            const auto v2 = _mm_extract_epi32(u32x4, 0);
            outd[i] = v2;
        }
    benchmark::DoNotOptimize(out);
}
BENCHMARK(f5);


static void f6(benchmark::State& state) {
    assert(SIZE % 4 == 0);

    std::vector<uint8_t> out(SIZE);

    const auto* ind = reinterpret_cast<const uint32_t*>(data.data());
    auto* outd = reinterpret_cast<uint32_t*>(out.data());

    for (auto _: state)
        for (std::size_t i=0; i < SIZE / 4; ++i) {
            const auto v = ind[i];
            __m128i u32x4 = _mm_cvtepu8_epi32(__m128i{v});
            __m128 fx4 = _mm_cvtepi32_ps(u32x4);
            fx4 = 255.0f * (fx4 / 128.0f);
            u32x4 = _mm_cvttps_epi32(fx4);
            u32x4 = _mm_packus_epi32(u32x4, u32x4);
            u32x4 = _mm_packus_epi16(u32x4, u32x4);

            const auto v2 = u32x4[0];
            outd[i] = v2;
        }
    benchmark::DoNotOptimize(out);
}
BENCHMARK(f6);


#include <immintrin.h>

static void f7(benchmark::State& state) {
    assert(SIZE % 8 == 0);

    std::vector<uint8_t> out(SIZE);

    const auto* ind = reinterpret_cast<const uint64_t*>(data.data());
    auto* outd = reinterpret_cast<uint64_t*>(out.data());

    for (auto _: state)
        for (std::size_t i=0; i < SIZE / 8; ++i) {
            const auto v = _mm_loadl_epi64(reinterpret_cast<const __m128i_u*>(ind + i));
            __m256i u32x8 = _mm256_cvtepu8_epi32(v);
            __m256 fx8 = _mm256_cvtepi32_ps(u32x8);
            fx8 = 255.0f * (fx8 / 128.0f);
            u32x8 = _mm256_cvttps_epi32(fx8);
            const __m256i u32x8_swap_hl = _mm256_permute2x128_si256(u32x8, u32x8, 0x01);
            const auto u16x16 = _mm256_packus_epi32(u32x8, u32x8_swap_hl);
            const auto u8x32 = _mm256_packus_epi16(u16x16, u16x16);
            outd[i] = u8x32[0];
        }
    benchmark::DoNotOptimize(out);
}
BENCHMARK(f7);


BENCHMARK_MAIN();
