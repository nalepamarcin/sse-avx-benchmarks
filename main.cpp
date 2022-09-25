#include <benchmark/benchmark.h>

#include <iostream>
#include <source_location>
#include <vector>

#include <cassert>
#include <cstdint>


#if defined __AVX512F__ && defined __AVX512VL__
#define ENABLE_AVX512_TESTS
#endif


#define VALIDATE_RESULTS
#define NORM_VAL 128.0f


constexpr const std::size_t SIZE = 2 << 20;
static const std::vector<uint8_t> data = []() {
    const uint8_t MAX_VAL = (int)NORM_VAL;
    std::vector<uint8_t> out(SIZE);
    for (std::size_t i=0; i < SIZE; ++i)
        out[i] = i % MAX_VAL;
    return out;
}();


#ifdef VALIDATE_RESULTS
static const std::vector<uint8_t> expected_data = []() {
    std::vector<uint8_t> out(SIZE);
    for (std::size_t i=0; i < SIZE; ++i)
        out[i] = 255.0f * (data[i] / NORM_VAL);
    return out;
}();


void validate_result(const std::vector<uint8_t>& result,
                     const std::source_location location = std::source_location::current()) {
    assert(result.size() == SIZE);
    for (std::size_t i=0; i < SIZE; ++i)
        if (result[i] != expected_data[i]) {
            std::cerr << "Data differs in test " << location.function_name() << " for i=" << i << ", found: " << (int)result[i] << " expected: " << (int)expected_data[i] << '\n';
            std::abort();
        }
}
#endif


static void vector_1u8_at_once_naive_direct(benchmark::State& state) {
    std::vector<uint8_t> out(SIZE);
    for (auto _: state)
        for (std::size_t i=0; i < SIZE; ++i) {
            out[i] = 255.0f * (data[i] / NORM_VAL);
        }
    benchmark::DoNotOptimize(out);
#ifdef VALIDATE_RESULTS
    validate_result(out);
#endif
}
BENCHMARK(vector_1u8_at_once_naive_direct);


static void vector_1u8_at_once_data_ptr(benchmark::State& state) {
    std::vector<uint8_t> out(SIZE);

    const uint8_t* ind = data.data();
    uint8_t* outd = out.data();

    for (auto _: state)
        for (std::size_t i=0; i < SIZE; ++i) {
            outd[i] = static_cast<uint8_t>(255.0f * (ind[i] / NORM_VAL));
        }
    benchmark::DoNotOptimize(out);
#ifdef VALIDATE_RESULTS
    validate_result(out);
#endif
}
BENCHMARK(vector_1u8_at_once_data_ptr);


#include <emmintrin.h>
#include <smmintrin.h>

static void sse_4u8_at_once_manual_recompose(benchmark::State& state) {
    assert(SIZE % 4 == 0);

    std::vector<uint8_t> out(SIZE);

    const auto* ind = reinterpret_cast<const uint32_t*>(data.data());
    auto* outd = reinterpret_cast<uint32_t*>(out.data());

    for (auto _: state)
        for (std::size_t i=0; i < SIZE / 4; ++i) {
            const auto v = ind[i];
            __m128i u32x4 = _mm_cvtepu8_epi32(__m128i{v});
            __m128 fx4 = _mm_cvtepi32_ps(u32x4);
            fx4 = 255.0f * (fx4 / NORM_VAL);
            u32x4 = _mm_cvttps_epi32(fx4);
            const auto v2 =
                    (u32x4[1] >> 32)  << 24 |
                    (u32x4[1] & 0xff) << 16 |
                    (u32x4[0] >> 32)  <<  8 |
                    (u32x4[0] & 0xff);
            outd[i] = v2;
        }
    benchmark::DoNotOptimize(out);
#ifdef VALIDATE_RESULTS
    validate_result(out);
#endif
}
BENCHMARK(sse_4u8_at_once_manual_recompose);


static void sse_4u8_at_once_extract_recompose(benchmark::State& state) {
    assert(SIZE % 4 == 0);

    std::vector<uint8_t> out(SIZE);

    const auto* ind = reinterpret_cast<const uint32_t*>(data.data());
    auto* outd = reinterpret_cast<uint32_t*>(out.data());

    for (auto _: state)
        for (std::size_t i=0; i < SIZE / 4; ++i) {
            const auto v = ind[i];
            __m128i u32x4 = _mm_cvtepu8_epi32(__m128i{v});
            __m128 fx4 = _mm_cvtepi32_ps(u32x4);
            fx4 = 255.0f * (fx4 / NORM_VAL);
            u32x4 = _mm_cvttps_epi32(fx4);
            const auto v2 =
                    _mm_extract_epi8(u32x4, 12) << 24 |
                    _mm_extract_epi8(u32x4,  8) << 16 |
                    _mm_extract_epi8(u32x4,  4) <<  8 |
                    _mm_extract_epi8(u32x4,  0);
            outd[i] = v2;
        }
    benchmark::DoNotOptimize(out);
#ifdef VALIDATE_RESULTS
    validate_result(out);
#endif
}
BENCHMARK(sse_4u8_at_once_extract_recompose);


static void sse_4u8_at_once_pack_recompose(benchmark::State& state) {
    assert(SIZE % 4 == 0);

    std::vector<uint8_t> out(SIZE);

    const auto* ind = reinterpret_cast<const uint32_t*>(data.data());
    auto* outd = reinterpret_cast<uint32_t*>(out.data());

    for (auto _: state)
        for (std::size_t i=0; i < SIZE / 4; ++i) {
            const auto v = ind[i];
            __m128i u32x4 = _mm_cvtepu8_epi32(__m128i{v});
            __m128 fx4 = _mm_cvtepi32_ps(u32x4);
            fx4 = 255.0f * (fx4 / NORM_VAL);
            u32x4 = _mm_cvttps_epi32(fx4);
            u32x4 = _mm_packus_epi32(u32x4, u32x4);
            u32x4 = _mm_packus_epi16(u32x4, u32x4);

            const auto v2 = _mm_extract_epi32(u32x4, 0);
            outd[i] = v2;
        }
    benchmark::DoNotOptimize(out);
#ifdef VALIDATE_RESULTS
    validate_result(out);
#endif
}
BENCHMARK(sse_4u8_at_once_pack_recompose);


#include <immintrin.h>

static void avx_8u8_at_once(benchmark::State& state) {
    assert(SIZE % 8 == 0);

    std::vector<uint8_t> out(SIZE);

    const auto* ind = reinterpret_cast<const uint64_t*>(data.data());
    auto* outd = reinterpret_cast<uint64_t*>(out.data());

    for (auto _: state)
        for (std::size_t i=0; i < SIZE / 8; ++i) {
            const auto v = _mm_loadl_epi64(reinterpret_cast<const __m128i_u*>(ind + i));
            __m256i u32x8 = _mm256_cvtepu8_epi32(v);
            __m256 fx8 = _mm256_cvtepi32_ps(u32x8);
            fx8 = 255.0f * (fx8 / NORM_VAL);
            u32x8 = _mm256_cvttps_epi32(fx8);
            const __m256i u32x8_swap_hl = _mm256_permute2x128_si256(u32x8, u32x8, 0x01);
            const auto u16x16 = _mm256_packus_epi32(u32x8, u32x8_swap_hl);
            const auto u8x32 = _mm256_packus_epi16(u16x16, u16x16);
            outd[i] = u8x32[0];
        }
    benchmark::DoNotOptimize(out);
#ifdef VALIDATE_RESULTS
    validate_result(out);
#endif
}
BENCHMARK(avx_8u8_at_once);


#ifdef ENABLE_AVX512_TESTS
static void avx_8u8_at_once_avx512_recompose(benchmark::State& state) {
    // not tested
    assert(SIZE % 8 == 0);

    std::vector<uint8_t> out(SIZE);

    const auto* ind = reinterpret_cast<const uint64_t*>(data.data());
    auto* outd = reinterpret_cast<uint64_t*>(out.data());

    for (auto _: state)
        for (std::size_t i=0; i < SIZE / 8; ++i) {
            const auto v = _mm_loadl_epi64(reinterpret_cast<const __m128i_u*>(ind + i));
            __m256i u32x8 = _mm256_cvtepu8_epi32(v);
            __m256 fx8 = _mm256_cvtepi32_ps(u32x8);
            fx8 = 255.0f * (fx8 / NORM_VAL);
            u32x8 = _mm256_cvttps_epi32(fx8);
            const __m128i u8x16 = _mm256_cvtepi32_epi8(u32x8);
            outd[i] = u8x16[0];
        }
    benchmark::DoNotOptimize(out);
#ifdef VALIDATE_RESULTS
    validate_result(out);
#endif
}
BENCHMARK(avx_8u8_at_once_avx512_recompose);
#endif


#ifdef ENABLE_AVX512_TESTS
static void avx_16u8_at_once_avx512(benchmark::State& state) {
    // not tested
    assert(SIZE % 16 == 0);

    std::vector<uint8_t> out(SIZE);

    const auto* ind = reinterpret_cast<const __uint128_t*>(data.data());
    auto* outd = reinterpret_cast<__uint128_t*>(out.data());

    for (auto _: state)
        for (std::size_t i=0; i < SIZE / 16; ++i) {
            const auto v = _mm_load_epi64(ind + i);
            __m512i u32x16 = _mm512_cvtepu8_epi32(v);
            __m512 fx16 = _mm512_cvtepu32_ps(u32x16);
            fx16 = 255.0f * (fx16 / NORM_VAL);
            u32x16 = _mm512_cvttps_epu32(fx16);
            const __m128i u8x16 = _mm512_cvtepi32_epi8(u32x16);
            outd[i] = (__uint128_t)u8x16;
        }
    benchmark::DoNotOptimize(out);
#ifdef VALIDATE_RESULTS
    validate_result(out);
#endif
}
BENCHMARK(avx_16u8_at_once_avx512);
#endif


BENCHMARK_MAIN();
