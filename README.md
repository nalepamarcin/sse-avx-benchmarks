# Description
Simple application for testing various methods of performing normalization of byte array.

Few implementations were provided, from simple, naive, direct vector access to SIMD with AVX-512.

# Building
Usual build of simple CMake project:
```
git submodule init
git submodule update
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

# Results
AMD Ryzen 5 5600X (no AVX-512 support):
```
Benchmark                                  Time             CPU   Iterations
----------------------------------------------------------------------------
vector_1u8_at_once_naive_direct      1112179 ns      1112081 ns          628
vector_1u8_at_once_data_ptr           299176 ns       299171 ns         2337
sse_4u8_at_once_manual_recompose      976459 ns       976418 ns          718
sse_4u8_at_once_extract_recompose     691740 ns       691714 ns         1013
sse_4u8_at_once_pack_recompose        336307 ns       335642 ns         2081
avx_8u8_at_once                       172303 ns       171876 ns         4081
```

Intel Core i5-1135G7 (with AVX-512):
```
Benchmark                                  Time             CPU   Iterations
----------------------------------------------------------------------------
vector_1u8_at_once_naive_direct      1264721 ns      1264730 ns          551
vector_1u8_at_once_data_ptr           253745 ns       253741 ns         2767
sse_4u8_at_once_manual_recompose     1030334 ns      1030328 ns          679
sse_4u8_at_once_extract_recompose    1044673 ns      1044664 ns          669
sse_4u8_at_once_pack_recompose        392863 ns       392858 ns         1784
avx_8u8_at_once                       291123 ns       291122 ns         2405
avx_8u8_at_once_avx512_recompose      171555 ns       171556 ns         4075
avx_16u8_at_once_avx512               135321 ns       135322 ns         5143
```
