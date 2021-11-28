use core::ops::RangeBounds;

use crate::{Rng, SeedError};

thread_local! {
    static RNG: Rng = Rng::from_seed_with_192bit([0x50b88b1357314d73, 0x8e4193b54c35742c, 0x47c6ff4737815fc5]);
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Seeds the thread local instance from the best available entropy pool.
pub fn seed() -> Result<(), SeedError> {
    RNG.with(|f| f.seed())
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Seeds the thread local instance with the given 64-bit seed.
pub fn seed_with_64bit(seed: u64) {
    RNG.with(|f| f.seed_with_64bit(seed))
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Seeds the thread local instance with the given 192-bit seed.
///
/// If the seed is of low quality, user should call [`mix()`] to improve the quality of the
/// first couple of random numbers.
///
/// # Notice
/// The variables must be seeded such that at least one bit of state is non-zero.
///
/// # Panics
/// Panics if all values are zero.
pub fn seed_with_192bit(seed: [u64; 3]) {
    RNG.with(|f| f.seed_with_192bit(seed))
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Mixes the state, which should improve the quality of the random numbers.
///
/// Should be called when having (re-)seeded the generator with a fixed value of low entropy.
pub fn mix() {
    RNG.with(|rng| rng.mix())
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random u8 value.
#[inline(always)]
pub fn u8() -> u8 {
    RNG.with(|rng| rng.u8())
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random u16 value.
#[inline(always)]
pub fn u16() -> u16 {
    RNG.with(|rng| rng.u16())
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random u32 value.
#[inline(always)]
pub fn u32() -> u32 {
    RNG.with(|rng| rng.u32())
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random u64 value.
#[inline(always)]
pub fn u64() -> u64 {
    RNG.with(|rng| rng.u64())
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random usize value.
#[inline(always)]
pub fn usize() -> usize {
    RNG.with(|rng| rng.usize())
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random i8 value.
#[inline(always)]
pub fn i8() -> i8 {
    RNG.with(|rng| rng.i8())
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random i16 value.
#[inline(always)]
pub fn i16() -> i16 {
    RNG.with(|rng| rng.i16())
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random i32 value.
#[inline(always)]
pub fn i32() -> i32 {
    RNG.with(|rng| rng.i32())
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random i64 value.
#[inline(always)]
pub fn i64() -> i64 {
    RNG.with(|rng| rng.i64())
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random isize value.
#[inline(always)]
pub fn isize() -> isize {
    RNG.with(|rng| rng.isize())
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random f32 value.
#[inline(always)]
pub fn f32() -> f32 {
    RNG.with(|rng| rng.f32())
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random f64 value.
#[inline(always)]
pub fn f64() -> f64 {
    RNG.with(|rng| rng.f64())
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random bool value.
#[inline(always)]
pub fn bool() -> bool {
    RNG.with(|rng| rng.bool())
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Randomly shuffles a slice.
pub fn shuffle<T>(slice: &mut [T]) {
    RNG.with(|rng| rng.shuffle(slice))
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Fills a mutable `[u8]` slice with random values.
pub fn fill_bytes(slice: &mut [u8]) {
    RNG.with(|rng| rng.fill_bytes(slice))
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random u8 value in the given range.
pub fn range_u8<T: RangeBounds<u8>>(range: T) -> u8 {
    RNG.with(|rng| rng.range_u8(range))
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random u16 value in the given range.
pub fn range_u16<T: RangeBounds<u16>>(range: T) -> u16 {
    RNG.with(|rng| rng.range_u16(range))
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random u32 value in the given range.
pub fn range_u32<T: RangeBounds<u32>>(range: T) -> u32 {
    RNG.with(|rng| rng.range_u32(range))
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random u64 value in the given range.
pub fn range_u64<T: RangeBounds<u64>>(range: T) -> u64 {
    RNG.with(|rng| rng.range_u64(range))
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random usize value in the given range.
pub fn range_usize<T: RangeBounds<usize>>(range: T) -> usize {
    RNG.with(|rng| rng.range_usize(range))
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random i8 value in the given range.
pub fn range_i8<T: RangeBounds<i8>>(range: T) -> i8 {
    RNG.with(|rng| rng.range_i8(range))
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random i16 value in the given range.
pub fn range_i16<T: RangeBounds<i16>>(range: T) -> i16 {
    RNG.with(|rng| rng.range_i16(range))
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random i32 value in the given range.
pub fn range_i32<T: RangeBounds<i32>>(range: T) -> i32 {
    RNG.with(|rng| rng.range_i32(range))
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random i64 value in the given range.
pub fn range_i64<T: RangeBounds<i64>>(range: T) -> i64 {
    RNG.with(|rng| rng.range_i64(range))
}

#[cfg_attr(docsrs, doc(cfg(feature = "tls")))]
/// Generates a random isize value in the given range.
pub fn range_isize<T: RangeBounds<isize>>(range: T) -> isize {
    RNG.with(|rng| rng.range_isize(range))
}