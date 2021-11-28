//! A pseudo random number generator using the [Romu](https://www.romu-random.org/) algorithm.
//!
//! This pseudo random number generator (PRNG) is not intended for cryptographic purposes. This
//! crate only implements the 64-bit "RomuTrio" generator, since it's the recommended generator
//! by the original author.
//!
//! ## Non-linear random number generator
//!
//! Romu is a non-linear random number generator. That means that the period is probabilistic and
//! is based on the seed. The bigger the needed period is, the higher the chance it is that the
//! actual period is "too small".
//!
//! Following formula is given by the author:
//!
//! ```ignore
//!     P(|cycle contains x<= 2^k|) = 2^k-s+7
//!         k is size of random numbers needed + 1.
//!         s is the state size.
//! ```
//!
//! Example chances for getting a "too small" period:
//!   * When 2^62 * 64 bit numbers are needed (32 EiB) -> 2^-122 chance
//!   * When 2^39 * 64 bit numbers are needed (4 TiB) -> 2^-146 chance
//!   * When 2^36 * 64 bit numbers are needed (512 GiB) -> 2^-149 chance
//!
//! You can read more about the theory behind Romu in the [official paper](https://arxiv.org/abs/2002.11331) and it's unique
//! selling points on the [official website](https://www.romu-random.org/) of the original author.
//!
//! ## Features
//!
//! The crate is `no_std` compatible.
//!
//!  * `std` - If `getrandom` is not used or returns an error, the generator will use the thread name and the current
//!            instance time to create a seed value. Enabled by default.
//!  * `tls` - Create static functions to use a thread local version of the generator. Enabled by default.
//!  * `getrandom` - Uses the `getrandom` crate to create a seed of high entropy. Enabled by default.
//!  * `unstable_tls` - Uses the unstable `thread_local` feature of Rust nightly. Improves the call times to the
//!                     thread local functions greatly.
//!  * `unstable_simd` - Uses the unstable `std::simd` crate of Rust nightly to provide special SIMD versions of the
//!                      generator which can be used to create large amount of random data fast.
#![warn(missing_docs)]
#![deny(clippy::unwrap_used)]
#![cfg_attr(feature = "unstable_tls", feature(thread_local))]
#![cfg_attr(feature = "unstable_simd", feature(portable_simd))]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(docsrs, feature(doc_cfg))]

#[cfg(feature = "unstable_simd")]
mod simd;
#[cfg(all(feature = "tls", not(feature = "unstable_tls")))]
mod stable_tls;
#[cfg(all(feature = "tls", feature = "unstable_tls"))]
mod unstable_tls;

use core::{
    cell::Cell,
    ops::{Bound, RangeBounds},
};

#[cfg(feature = "unstable_simd")]
pub use simd::*;
#[cfg(all(feature = "tls", not(feature = "unstable_tls")))]
pub use stable_tls::*;
#[cfg(all(feature = "tls", feature = "unstable_tls"))]
pub use unstable_tls::*;

macro_rules! range_integer {
    ($fn:tt, $target:tt, $base:tt, $tmp:tt, $doc:tt) => {
        #[doc = $doc]
        pub fn $fn<T: RangeBounds<$target>>(&self, range: T) -> $target {
            let low = match range.start_bound() {
                Bound::Included(&x) => x,
                Bound::Excluded(&x) => x.checked_add(1).unwrap_or_else(|| {
                    panic!(
                        "start is invalid: {:?}..{:?}",
                        range.start_bound(),
                        range.end_bound()
                    )
                }),
                Bound::Unbounded => $target::MIN,
            };

            let high = match range.end_bound() {
                Bound::Included(&x) => x,
                Bound::Excluded(&x) => x.checked_sub(1).unwrap_or_else(|| {
                    panic!(
                        "end is invalid: {:?}..{:?}",
                        range.start_bound(),
                        range.end_bound()
                    )
                }),
                Bound::Unbounded => $target::MAX,
            };

            if low > high {
                panic!(
                    "start is bigger than end: {:?}..{:?}",
                    range.start_bound(),
                    range.end_bound()
                );
            }

            if low == $target::MIN && high == $target::MAX {
                self.next() as $target
            } else {
                let s = high.wrapping_sub(low).wrapping_add(1) as $base;

                // As described in "Fast Random Integer Generation in an Interval" by Daniel Lemire.
                // <https://arxiv.org/abs/1805.10941>
                let mut x = self.next() as $base;
                let mut m = (x as $tmp).wrapping_mul(s as $tmp);
                let mut l = m as $base;
                if l < s {
                    let t = s.wrapping_neg() % s;
                    while l < t {
                        x = self.next() as $base;
                        m = (x as $tmp).wrapping_mul(s as $tmp);
                        l = m as $base;
                    }
                }

                low.wrapping_add((m >> (core::mem::size_of::<$base>() * 8)) as $target)
            }
        }
    };
}

/// Error thrown when no seed could be created from an entropy source.
#[derive(Clone, PartialEq, Debug)]
pub struct SeedError(String);

impl core::fmt::Display for SeedError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "can't create seed: {}", &self.0)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SeedError {}

fn generate_seed() -> Result<[u64; 3], SeedError> {
    #[cfg(feature = "getrandom")]
    return collect_getrandom_entropy();
    #[cfg(all(feature = "std", not(feature = "getrandom")))]
    return collect_std_entropy();
    #[cfg(all(not(feature = "std"), not(feature = "getrandom")))]
    return Err(SeedError("no entropy source available".to_string()));
}

#[cfg(feature = "getrandom")]
#[allow(unused_variables)]
fn collect_getrandom_entropy() -> Result<[u64; 3], SeedError> {
    let mut b = [0u8; 24];
    match getrandom::getrandom(&mut b) {
        Ok(_) => Ok([
            // Full 192 bit state from high quality source.
            u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]),
            u64::from_be_bytes([b[8], b[9], b[10], b[11], b[12], b[13], b[14], b[15]]),
            u64::from_be_bytes([b[16], b[17], b[18], b[19], b[20], b[21], b[22], b[23]]),
        ]),
        Err(err) => {
            // Reduced 64 bit state from low quality source as fallback.
            #[cfg(feature = "std")]
            return collect_std_entropy();
            #[cfg(not(feature = "std"))]
            return Err(SeedError(format!(
                "getrandom error and no fallback available: {}",
                err
            )));
        }
    }
}

#[cfg(feature = "std")]
fn collect_std_entropy() -> Result<[u64; 3], SeedError> {
    use std::{
        collections::hash_map::DefaultHasher,
        hash::{Hash, Hasher},
        thread,
        time::Instant,
    };

    let mut hasher = DefaultHasher::new();
    Instant::now().hash(&mut hasher);
    thread::current().id().hash(&mut hasher);

    Ok(split_mix_64(hasher.finish()))
}

// We use `SplitMix64` to "improve" the seed as suggested by the author of Romu.
const fn split_mix_64(state: u64) -> [u64; 3] {
    let (x, state) = split_mix_64_inner(state);
    let (y, state) = split_mix_64_inner(state);
    let (z, _) = split_mix_64_inner(state);

    [x, y, z]
}

/// Written in 2015 by Sebastiano Vigna (vigna@acm.org).
///
/// To the extent possible under law, the author has dedicated all copyright
/// and related and neighboring rights to this software to the public domain
/// worldwide. This software is distributed without any warranty.
///
/// See <http://creativecommons.org/publicdomain/zero/1.0/>.
const fn split_mix_64_inner(mut state: u64) -> (u64, u64) {
    state += 0x9E3779B97F4A7C15;

    let mut z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB;
    z = z ^ (z >> 31);

    (z, state)
}

/// Implements `RomuTrio`.
pub struct Rng {
    x: Cell<u64>,
    y: Cell<u64>,
    z: Cell<u64>,
}

impl Rng {
    /// Creates a new [`Rng`] with a seed from the best available entropy pool.
    pub fn new() -> Result<Self, SeedError> {
        let rng = Self {
            x: Cell::new(0),
            y: Cell::new(0),
            z: Cell::new(0),
        };
        rng.seed()?;

        Ok(rng)
    }

    /// Creates a new [`Rng`] from the given 64-bit seed.
    pub fn from_seed_with_64bit(seed: u64) -> Self {
        let seed = split_mix_64(seed);

        Self {
            x: Cell::new(seed[0] | 1),
            y: Cell::new(seed[1] | 1),
            z: Cell::new(seed[2] | 1),
        }
    }

    /// Creates a new [`Rng`] from the given 192-bit seed.     
    ///
    /// If the seed is of low quality, user should call [`Rng::mix()`] to improve the quality of the
    /// first couple of random numbers.
    ///
    /// # Notice
    /// The variables must be seeded such that at least one bit of state is non-zero.
    ///
    /// # Panics
    /// Panics if all values are zero.
    pub fn from_seed_with_192bit(seed: [u64; 3]) -> Self {
        assert!(seed[0] != 0 && seed[1] != 0 && seed[2] != 0, "seed it zero");

        Self {
            x: Cell::new(seed[0]),
            y: Cell::new(seed[1]),
            z: Cell::new(seed[2]),
        }
    }

    /// Re-seeds the [`Rng`] from the best available entropy pool.
    ///
    /// Returns `False` if no seed with good entropy could be generated.
    pub fn seed(&self) -> Result<(), SeedError> {
        let seed = generate_seed()?;
        self.x.set(seed[0]);
        self.y.set(seed[1]);
        self.z.set(seed[2]);

        Ok(())
    }

    /// Re-seeds the [`Rng`] with the given 64-bit seed.
    pub fn seed_with_64bit(&self, seed: u64) {
        let seed = split_mix_64(seed);

        self.x.set(seed[0]);
        self.y.set(seed[1]);
        self.z.set(seed[2]);
    }

    /// Re-seeds the [`Rng`] from the given 192-bit seed.
    ///
    /// If the seed is of low quality, user should call [`Rng::mix()`] to improve the quality of the
    /// first couple of random numbers.
    ///
    /// # Notice
    /// The variables must be seeded such that at least one bit of state is non-zero.
    ///
    /// # Panics
    /// Panics if all values are zero.
    pub fn seed_with_192bit(&self, seed: [u64; 3]) {
        assert!(seed[0] != 0 && seed[1] != 0 && seed[2] != 0, "seed is zero");

        self.x.set(seed[0]);
        self.y.set(seed[1]);
        self.z.set(seed[2]);
    }

    /// Mixes the state, which should improve the quality of the random numbers.
    ///
    /// Should be called when having (re-)seeded the generator with a fixed value of low entropy.
    pub fn mix(&self) {
        (0..10).into_iter().for_each(|_| {
            self.next();
        });
    }

    /// The actual `RomuTrio` algorithm.
    ///
    /// Great for general purpose work, including huge jobs.
    /// Est. capacity = 2^75 bytes. State size = 192 bits.
    ///
    /// Copyright 2020 Mark A. Overton
    /// Licensed under Apache-2.0.
    #[inline(always)]
    fn next(&self) -> u64 {
        let xp = self.x.get();
        let yp = self.y.get();
        let zp = self.z.get();

        self.x.set(zp.wrapping_mul(0xD3833E804F4C574B));
        let y = yp.wrapping_sub(xp);
        self.y.set(y.wrapping_shl(12) | y.wrapping_shr(52));
        let z = zp.wrapping_sub(yp);
        self.z.set(z.wrapping_shl(44) | z.wrapping_shr(20));

        xp
    }

    /// Generates a random u8 value.
    #[inline(always)]
    pub fn u8(&self) -> u8 {
        self.next() as u8
    }

    /// Generates a random u16 value.
    #[inline(always)]
    pub fn u16(&self) -> u16 {
        self.next() as u16
    }

    /// Generates a random u32 value.
    #[inline(always)]
    pub fn u32(&self) -> u32 {
        self.next() as u32
    }

    /// Generates a random u64 value.
    #[inline(always)]
    pub fn u64(&self) -> u64 {
        self.next()
    }

    /// Generates a random usize value.
    #[inline(always)]
    pub fn usize(&self) -> usize {
        self.next() as usize
    }

    /// Generates a random i8 value.
    #[inline(always)]
    pub fn i8(&self) -> i8 {
        i8::from_ne_bytes(self.u8().to_ne_bytes())
    }

    /// Generates a random i16 value.
    #[inline(always)]
    pub fn i16(&self) -> i16 {
        i16::from_ne_bytes(self.u16().to_ne_bytes())
    }

    /// Generates a random i32 value.
    #[inline(always)]
    pub fn i32(&self) -> i32 {
        i32::from_ne_bytes(self.u32().to_ne_bytes())
    }

    /// Generates a random i64 value.
    #[inline(always)]
    pub fn i64(&self) -> i64 {
        i64::from_ne_bytes(self.u64().to_ne_bytes())
    }

    /// Generates a random isize value.
    #[inline(always)]
    pub fn isize(&self) -> isize {
        isize::from_ne_bytes(self.usize().to_ne_bytes())
    }

    /// Generates a random bool value.
    #[inline(always)]
    pub fn bool(&self) -> bool {
        self.next() % 2 == 0
    }

    /// Generates a random f32 value in range (0..1).
    #[inline(always)]
    pub fn f32(&self) -> f32 {
        #[cfg(feature = "std")]
        return ((self.u32() >> 8) as f32) * (f32::exp2(-24.0));
        #[cfg(not(feature = "std"))]
        return ((self.u32() >> 8) as f32) * 0.000000059604645;
    }

    /// Generates a random f64 value in range (0..1).
    #[inline(always)]
    pub fn f64(&self) -> f64 {
        #[cfg(feature = "std")]
        return ((self.u64() >> 11) as f64) * (f64::exp2(-53.0));
        #[cfg(not(feature = "std"))]
        return ((self.u64() >> 11) as f64) * 0.00000000000000011102230246251565;
    }

    /// Randomly shuffles a slice.
    pub fn shuffle<T>(&self, slice: &mut [T]) {
        for i in 1..slice.len() {
            slice.swap(i, self.range_usize(..=i));
        }
    }

    /// Fills a mutable `[u8]` slice with random values.
    pub fn fill_bytes(&self, slice: &mut [u8]) {
        let mut chunks = slice.chunks_exact_mut(8);
        for chunk in &mut chunks {
            chunk.copy_from_slice(&self.next().to_ne_bytes())
        }
        chunks
            .into_remainder()
            .iter_mut()
            .for_each(|x| *x = self.next() as u8);
    }

    range_integer!(
        range_u8,
        u8,
        u8,
        u16,
        "Generates a random u8 value in the given range."
    );

    range_integer!(
        range_u16,
        u16,
        u16,
        u32,
        "Generates a random u16 value in the given range."
    );

    range_integer!(
        range_u32,
        u32,
        u32,
        u64,
        "Generates a random u32 value in the given range."
    );

    range_integer!(
        range_u64,
        u64,
        u64,
        u128,
        "Generates a random u64 value in the given range."
    );

    #[cfg(target_pointer_width = "16")]
    range_integer!(
        range_usize,
        usize,
        u16,
        u32,
        "Generates a random usize value in the given range."
    );

    #[cfg(target_pointer_width = "32")]
    range_integer!(
        range_usize,
        usize,
        u32,
        u64,
        "Generates a random usize value in the given range."
    );

    #[cfg(target_pointer_width = "64")]
    range_integer!(
        range_usize,
        usize,
        u64,
        u128,
        "Generates a random usize value in the given range."
    );

    range_integer!(
        range_i8,
        i8,
        u8,
        u16,
        "Generates a random i8 value in the given range."
    );

    range_integer!(
        range_i16,
        i16,
        u16,
        u32,
        "Generates a random i16 value in the given range."
    );

    range_integer!(
        range_i32,
        i32,
        u32,
        u64,
        "Generates a random i32 value in the given range."
    );

    range_integer!(
        range_i64,
        i64,
        u64,
        u128,
        "Generates a random i64 value in the given range."
    );

    #[cfg(target_pointer_width = "16")]
    range_integer!(
        range_isize,
        isize,
        u16,
        u32,
        "Generates a random isize value in the given range."
    );

    #[cfg(target_pointer_width = "32")]
    range_integer!(
        range_isize,
        isize,
        u32,
        u64,
        "Generates a random isize value in the given range."
    );

    #[cfg(target_pointer_width = "64")]
    range_integer!(
        range_isize,
        isize,
        u64,
        u128,
        "Generates a random isize value in the given range."
    );
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used)]
    use super::*;

    #[test]
    fn test_scalar() {
        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert_eq!(rng.u8(), 226);
        assert_eq!(rng.u8(), 92);

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert_eq!(rng.u16(), 64738);
        assert_eq!(rng.u16(), 50524);

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert_eq!(rng.u32(), 2204433634);
        assert_eq!(rng.u32(), 3535914332);

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert_eq!(rng.u64(), 3296835985448697058);
        assert_eq!(rng.u64(), 4696255203626829148);

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert_eq!(rng.usize(), 3296835985448697058);
        assert_eq!(rng.usize(), 4696255203626829148);

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert!((rng.f32() - 0.51325965).abs() < f32::EPSILON);
        assert!((rng.f32() - 0.8232692).abs() < f32::EPSILON);

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert!((rng.f64() - 0.17872183688759324).abs() < f64::EPSILON);
        assert!((rng.f64() - 0.2545845047159281).abs() < f64::EPSILON);

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert!(rng.bool());
        assert!(rng.bool());
        assert!(!rng.bool());
    }

    #[test]
    fn test_range() {
        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert_eq!(rng.range_u8(0..128), 113);
        assert_eq!(rng.range_u8(0..128), 46);

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert_eq!(rng.range_u16(0..128), 126);
        assert_eq!(rng.range_u16(0..128), 98);

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert_eq!(rng.range_u32(0..128), 65);
        assert_eq!(rng.range_u32(0..128), 105);

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert_eq!(rng.range_u64(0..128), 22);
        assert_eq!(rng.range_u64(0..128), 32);

        let rng = Rng::from_seed_with_192bit([u64::MAX; 3]);
        rng.mix();
        assert_eq!(rng.range_usize(0..128), 75);
        assert_eq!(rng.range_usize(0..128), 99);

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert_eq!(rng.range_i8(-64..64), 49);
        assert_eq!(rng.range_i8(-64..64), -18);

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert_eq!(rng.range_i16(-64..64), 62);
        assert_eq!(rng.range_i16(-64..64), 34);

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert_eq!(rng.range_i32(-64..64), 1);
        assert_eq!(rng.range_i32(-64..64), 41);

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert_eq!(rng.range_i64(-64..64), -42);
        assert_eq!(rng.range_i64(-64..64), -32);

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.mix();
        assert_eq!(rng.range_isize(-64..64), -42);
        assert_eq!(rng.range_isize(-64..64), -32);
    }

    #[test]
    fn test_shuffle() {
        let mut slice = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let org_slice = slice;

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.shuffle(&mut slice);

        assert_ne!(org_slice, slice);
    }

    #[test]
    fn test_fill_bytes() {
        let mut bytes = [1u8; 301];
        let org_bytes = bytes;

        let rng = Rng::from_seed_with_192bit([42; 3]);
        rng.fill_bytes(&mut bytes);

        assert_ne!(org_bytes, bytes);
    }
}
