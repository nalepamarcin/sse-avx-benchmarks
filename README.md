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
Results on AMD Ryzen 5 5600X (no AVX-512 support):
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
