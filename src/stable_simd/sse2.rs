#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use core::mem::{size_of, transmute};

use crate::{generate_seed, split_mix_64, SeedSource};

/// Implements `RomuTrio` with 512-bit width.
pub struct RngWide {
    x: [__m128i; 4],
    y: [__m128i; 4],
    z: [__m128i; 4],
    seed_source: SeedSource,
}

impl Default for RngWide {
    fn default() -> Self {
        unsafe {
            let mut rng = Self {
                x: [
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                ],
                y: [
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                ],
                z: [
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                    _mm_setzero_si128(),
                ],
                seed_source: SeedSource::Fixed,
            };
            rng.seed();
            rng
        }
    }
}

impl RngWide {
    /// Creates a new [`RngWide`] with a seed from the best available randomness source.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new [`RngWide`] from the given eight 64-bit seeds.
    ///
    /// The seeds should be from a high randomness source.
    pub const fn from_seed_with_64bit(seeds: [u64; 8]) -> Self {
        let lane0 = split_mix_64(seeds[0]);
        let lane1 = split_mix_64(seeds[1]);
        let lane2 = split_mix_64(seeds[2]);
        let lane3 = split_mix_64(seeds[3]);
        let lane4 = split_mix_64(seeds[4]);
        let lane5 = split_mix_64(seeds[5]);
        let lane6 = split_mix_64(seeds[6]);
        let lane7 = split_mix_64(seeds[7]);

        assert!(size_of::<[__m128i; 4]>() == size_of::<[[u64; 2]; 4]>());
        unsafe {
            Self {
                x: transmute([
                    [lane0[0], lane1[0]],
                    [lane2[0], lane3[0]],
                    [lane4[0], lane5[0]],
                    [lane6[0], lane7[0]],
                ]),
                y: transmute([
                    [lane0[1], lane1[1]],
                    [lane2[1], lane3[1]],
                    [lane4[1], lane5[1]],
                    [lane6[1], lane7[1]],
                ]),
                z: transmute([
                    [lane0[2], lane1[2]],
                    [lane2[2], lane3[2]],
                    [lane4[2], lane5[2]],
                    [lane6[2], lane7[2]],
                ]),
                seed_source: SeedSource::User,
            }
        }
    }

    /// Creates a new [`RngWide`] from the given eight 192-bit seeds.
    ///
    /// The seeds should be from a high randomness source.
    ///
    /// If the seeds are of low quality, user should call [`RngWide::mix()`] to improve the quality of the
    /// first couple of random numbers.
    ///
    /// # Notice
    /// The variables must be seeded such that at least one bit of state is non-zero.
    pub const fn from_seed_with_192bit(seeds: [[u64; 3]; 8]) -> Self {
        assert!(size_of::<[__m128i; 4]>() == size_of::<[[u64; 2]; 4]>());
        unsafe {
            Self {
                x: transmute([
                    [seeds[0][0], seeds[1][0]],
                    [seeds[2][0], seeds[3][0]],
                    [seeds[4][0], seeds[5][0]],
                    [seeds[6][0], seeds[7][0]],
                ]),
                y: transmute([
                    [seeds[0][1], seeds[1][1]],
                    [seeds[2][1], seeds[3][1]],
                    [seeds[4][1], seeds[5][1]],
                    [seeds[6][1], seeds[7][1]],
                ]),
                z: transmute([[
                    [seeds[0][2], seeds[1][2]],
                    [seeds[2][2], seeds[3][2]],
                    [seeds[4][2], seeds[5][2]],
                    [seeds[6][2], seeds[7][2]],
                ]]),
                seed_source: SeedSource::User,
            }
        }
    }

    /// Mixes the states, which should improve the quality of the random numbers.
    ///
    /// Should be called when having (re-)seeded the generator with a fixed value of low randomness.
    pub fn mix(&mut self) {
        (0..10).for_each(|_| {
            self.next();
        });
    }

    /// Re-seeds the [`RngWide`] from the best available randomness source.
    pub fn seed(&mut self) {
        let mut memory_address = self as *const _ as u64;
        let mut seed_source = SeedSource::Fixed;

        let mut x = [0u64; 8];
        let mut y = [0u64; 8];
        let mut z = [0u64; 8];

        x.iter_mut()
            .zip(y.iter_mut())
            .zip(z.iter_mut())
            .for_each(|((x, y), z)| {
                let (lane, source) = generate_seed(memory_address);

                *x = lane[0];
                *y = lane[1];
                *z = lane[2];

                seed_source = source;
                memory_address = memory_address.wrapping_add(1);
            });

        assert!(size_of::<[__m128i; 4]>() == size_of::<[u64; 8]>());
        unsafe {
            self.x = transmute(x);
            self.y = transmute(y);
            self.z = transmute(z);
        }

        self.seed_source = seed_source;
    }

    /// The actual wide `RomuTrio` algorithm.
    ///
    /// Great for general purpose work, including huge jobs.
    /// Est. capacity = 2^75 bytes. State size = 192 bits.
    ///
    /// Copyright 2020 Mark A. Overton
    /// Licensed under Apache-2.0.
    #[inline(always)]
    fn next(&mut self) -> [u64; 8] {
        let xp = self.x;
        let yp = self.y;
        let zp = self.z;

        unsafe {
            // 0xD3833E804F4C574B
            let high_mul = _mm_set1_epi64x(3548593792);
            let low_mul = _mm_set1_epi64x(1330403147);

            let zp_high = _mm_mul_epu32(zp[0], high_mul);
            let zp_high_shift = _mm_srli_epi64::<32>(zp[0]);
            let zp_mid = _mm_mul_epu32(zp_high_shift, low_mul);
            let zp_mid_high = _mm_add_epi64(zp_mid, zp_high);
            let zp_mid_high = _mm_slli_epi64::<32>(zp_mid_high);
            let zp_low = _mm_mul_epu32(low_mul, zp[0]);
            let x0 = _mm_add_epi64(zp_low, zp_mid_high);

            let zp_high = _mm_mul_epu32(zp[1], high_mul);
            let zp_high_shift = _mm_srli_epi64::<32>(zp[1]);
            let zp_mid = _mm_mul_epu32(zp_high_shift, low_mul);
            let zp_mid_high = _mm_add_epi64(zp_mid, zp_high);
            let zp_mid_high = _mm_slli_epi64::<32>(zp_mid_high);
            let zp_low = _mm_mul_epu32(low_mul, zp[1]);
            let x1 = _mm_add_epi64(zp_low, zp_mid_high);

            let zp_high = _mm_mul_epu32(zp[2], high_mul);
            let zp_high_shift = _mm_srli_epi64::<32>(zp[2]);
            let zp_mid = _mm_mul_epu32(zp_high_shift, low_mul);
            let zp_mid_high = _mm_add_epi64(zp_mid, zp_high);
            let zp_mid_high = _mm_slli_epi64::<32>(zp_mid_high);
            let zp_low = _mm_mul_epu32(low_mul, zp[2]);
            let x2 = _mm_add_epi64(zp_low, zp_mid_high);

            let zp_high = _mm_mul_epu32(zp[3], high_mul);
            let zp_high_shift = _mm_srli_epi64::<32>(zp[3]);
            let zp_mid = _mm_mul_epu32(zp_high_shift, low_mul);
            let zp_mid_high = _mm_add_epi64(zp_mid, zp_high);
            let zp_mid_high = _mm_slli_epi64::<32>(zp_mid_high);
            let zp_low = _mm_mul_epu32(low_mul, zp[3]);
            let x3 = _mm_add_epi64(zp_low, zp_mid_high);

            self.x = [x0, x1, x2, x3];

            let ty = _mm_sub_epi64(yp[0], xp[0]);
            let srl = _mm_srli_epi64::<52>(ty);
            let sll = _mm_slli_epi64::<12>(ty);
            let y0 = _mm_or_si128(sll, srl);

            let ty = _mm_sub_epi64(yp[1], xp[1]);
            let srl = _mm_srli_epi64::<52>(ty);
            let sll = _mm_slli_epi64::<12>(ty);
            let y1 = _mm_or_si128(sll, srl);

            let ty = _mm_sub_epi64(yp[2], xp[2]);
            let srl = _mm_srli_epi64::<52>(ty);
            let sll = _mm_slli_epi64::<12>(ty);
            let y2 = _mm_or_si128(sll, srl);

            let ty = _mm_sub_epi64(yp[3], xp[3]);
            let srl = _mm_srli_epi64::<52>(ty);
            let sll = _mm_slli_epi64::<12>(ty);
            let y3 = _mm_or_si128(sll, srl);

            self.y = [y0, y1, y2, y3];

            let tz = _mm_sub_epi64(zp[0], yp[0]);
            let srl = _mm_srli_epi64::<20>(tz);
            let sll = _mm_slli_epi64::<44>(tz);
            let z0 = _mm_or_si128(sll, srl);

            let tz = _mm_sub_epi64(zp[1], yp[1]);
            let srl = _mm_srli_epi64::<20>(tz);
            let sll = _mm_slli_epi64::<44>(tz);
            let z1 = _mm_or_si128(sll, srl);

            let tz = _mm_sub_epi64(zp[2], yp[2]);
            let srl = _mm_srli_epi64::<20>(tz);
            let sll = _mm_slli_epi64::<44>(tz);
            let z2 = _mm_or_si128(sll, srl);

            let tz = _mm_sub_epi64(zp[3], yp[3]);
            let srl = _mm_srli_epi64::<20>(tz);
            let sll = _mm_slli_epi64::<44>(tz);
            let z3 = _mm_or_si128(sll, srl);

            self.z = [z0, z1, z2, z3];

            assert!(size_of::<[__m128i; 4]>() == size_of::<[u64; 8]>());
            transmute(xp)
        }
    }

    /// Generates eight random u64 values.
    #[inline(always)]
    pub fn u64x8(&mut self) -> [u64; 8] {
        self.next()
    }

    /// Fills a mutable `[u8]` slice with random values.
    #[allow(clippy::manual_bits)]
    pub fn fill_bytes(&mut self, slice: &mut [u8]) {
        const CHUNK_SIZE: usize = 8 * size_of::<u64>();

        let mut chunks = slice.chunks_exact_mut(CHUNK_SIZE);
        for chunk in &mut chunks {
            let data: [u8; CHUNK_SIZE] = unsafe { transmute(self.next()) };
            chunk.copy_from_slice(&data);
        }

        assert!(size_of::<[u8; CHUNK_SIZE]>() == size_of::<[u64; 8]>());
        let data: [u8; CHUNK_SIZE] = unsafe { transmute(self.next()) };
        chunks
            .into_remainder()
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = data[i]);
    }
}
