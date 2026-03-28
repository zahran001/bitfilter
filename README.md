# BitFilter

[![CI](https://github.com/zahran001/bitfilter/actions/workflows/ci.yml/badge.svg)](https://github.com/zahran001/bitfilter/actions/workflows/ci.yml)

A SIMD-accelerated audience segmentation engine in C++20.

Answers boolean bitmap queries (`A AND B AND NOT C`) over 500M users at memory-bandwidth speed using AVX2 (x86) and SVE (ARM).
