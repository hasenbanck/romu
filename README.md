# romu

[![Documentation](https://docs.rs/romu/badge.svg)](https://docs.rs/romu/)
[![Crates.io](https://img.shields.io/crates/v/romu.svg)](https://crates.io/crates/romu)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE-APACHE)

A pseudo random number generator using the algorithm [Romu](https://www.romu-random.org/) for the
programing language Rust.

This pseudo random number generator (PRNG) is not intended for cryptographic purposes. This crate only implements the
64-bit "RomuTrio" generator, since it's the recommended generator by the original author.

## Non-linear random number generator

Romu is a non-linear random number generator. That means that the period is probabilistic and is based on the seed.
The bigger the needed period is, the higher the chance it is that the actual period is "too small".

Following formula is given by the author:

```
    P(|cycle contains x<= 2^k|) = 2^k-s+7
        k is size of random numbers needed + 1.
        s is the state size.
```

Example chances for getting a "too small" period:
 * When 2^62 * 64-bit numbers are needed (32 EiB) -> 2^-122 chance
 * When 2^39 * 64-bit numbers are needed (4 TiB) -> 2^-146 chance
 * When 2^36 * 64-bit numbers are needed (512 GiB) -> 2^-149 chance

You can read more about the theory behind Romu in the [official paper](https://arxiv.org/abs/2002.11331) and it's unique
selling points on the [official website](https://www.romu-random.org/) of the original author.

## Seeding

When the user calls the `new()` or `default()` functions of a generator, the implementation
tries to use the best available randomness source to seed the generator (in the following order):
 1. The crate `getrandom` to seed from a high quality randomness source of the operating system.
    The feature `getrandom` must be activated for this.
 2. Use the functionality of the standard library to create a low quality randomness seed (using
    the current time, the thread ID and a memory address).
    The feature `std` must be activated for this.
 3. Use a memory address as a very low randomness seed. If Address Space Layout Randomization
    (ASLR) is supported by the operating system, this should be a pretty "random" value.

It is highly recommended using the `no_std` compatible `getrandom` feature to get high quality
randomness seeds.

The user can always create / update a generator with a user provided seed value.

If the `tls` feature is used, the user should call the `seed()` function to seed the TLS
before creating the first random numbers, since the TLS instance is instantiated with a fixed
value.

## SIMD

The crate currently provides three wide generators that try to speed up the generation for large amount
of random numbers.

 * `Rng128` - Generator with a width of 128-bit.
 * `Rng256` - Generator with a width of 256-bit.
 * `Rng512` - Generator with a width of 512-bit.
 
The fallbacks will not produce auto vectorized code and are only useful to better utilize the CPU pipeline.
Using `Rng128` will result in a throughput improvement when filling a byte slice though (AMD Ryzen 9 5950X):

```
bytes/Rng/1048576       time:   [75.418 us 75.578 us 75.751 us]
                        thrpt:  [12.892 GiB/s 12.921 GiB/s 12.949 GiB/s]
bytes/Rng128/1048576    time:   [57.103 us 57.384 us 57.711 us]
                        thrpt:  [16.921 GiB/s 17.018 GiB/s 17.102 GiB/s]
```

Hand-written SSE2 generators don't improve the throughput since they can't calculate a 64-bit multiplication
in one instruction and need multiple 32-bit multiplications instead and are not wide enough to compensate.

AVX2 though has special hand-written implementations for `Rng256` and `Rng512` which greatly improve the
throughput (AMD Ryzen 9 5950X):

```
bytes/Rng/1048576       time:   [75.774 us 76.037 us 76.313 us]
                        thrpt:  [12.797 GiB/s 12.843 GiB/s 12.888 GiB/s]
bytes/Rng128/1048576    time:   [56.840 us 57.031 us 57.251 us]
                        thrpt:  [17.057 GiB/s 17.123 GiB/s 17.181 GiB/s]
bytes/Rng256/1048576    time:   [54.443 us 54.562 us 54.680 us]
                        thrpt:  [17.859 GiB/s 17.898 GiB/s 17.937 GiB/s]
bytes/Rng512/1048576    time:   [35.528 us 35.618 us 35.721 us]
                        thrpt:  [27.338 GiB/s 27.418 GiB/s 27.487 GiB/s]
```

The nightly only feature `unstable_simd` uses the `core::simd` create to implement the SIMD generators.

## Features

The crate is `no_std` compatible.

 * `std` - If `getrandom` is not used or returns an error, the generator will use the thread name and the current
           instance time to create a seed value. Enabled by default.
 * `tls` - Creates static functions that use a thread local version of the generator. Enabled by default.
 * `getrandom` - Uses the `getrandom` crate to create a seed of high randomness. Enabled by default.
 * `unstable_tls` - Uses the unstable `thread_local` feature of Rust nightly. Improves the call times to the
                    thread local functions greatly. 
 * `unstable_simd` - Uses the unstable `std::simd` crate of Rust nightly to provide the SIMD versions of the wide
                     generators.

## License

Licensed under Apache License, Version 2.0, ([LICENSE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as
defined in the Apache-2.0 license without any additional terms or conditions.
