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

The crate provides a wide generator that tries to speed up the generation for large amount of random numbers by trying
to utilize SIMD instructions.

Handwritten SSE2 and AVX2 implementations are available. A fallback is provided but won't produce auto
vectorized code.

The nightly only feature `unstable_simd` uses the `core::simd` crate to implement the wide generator.

SSE2 benchmark:
```
bytes/Rng/1048576       time:   [73.299 µs 73.359 µs 73.437 µs]
                        thrpt:  [13.298 GiB/s 13.312 GiB/s 13.323 GiB/s]
bytes/RngWide/1048576   time:   [63.837 µs 63.998 µs 64.245 µs]
                        thrpt:  [15.201 GiB/s 15.259 GiB/s 15.298 GiB/s]
```

AVX2 benchmark:
```
bytes/Rng/1048576       time:   [74.062 µs 74.253 µs 74.519 µs]
                        thrpt:  [13.105 GiB/s 13.152 GiB/s 13.186 GiB/s]
bytes/RngWide/1048576   time:   [35.187 µs 35.266 µs 35.375 µs]
                        thrpt:  [27.606 GiB/s 27.691 GiB/s 27.754 GiB/s]
```

## Features

The crate is `no_std` compatible.

 * `std` - If `getrandom` is not used or returns an error, the generator will use the thread name and the current
           instance time to create a seed value. Enabled by default.
 * `tls` - Creates static functions that use a thread local version of the generator. Enabled by default.
 * `getrandom` - Uses the `getrandom` crate to create a seed of high randomness. Enabled by default.
 * `unstable_tls` - Uses the unstable `thread_local` feature of Rust nightly. Improves the call times to the
                    thread local functions greatly. 
 * `unstable_simd` - Uses the unstable `std::simd` crate of Rust nightly to provide the SIMD version of the wide
                     generator.

## License

Licensed under Apache License, Version 2.0, ([LICENSE](LICENSE) or http://www.apache.org/licenses/LICENSE-2.0).

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as
defined in the Apache-2.0 license without any additional terms or conditions.
