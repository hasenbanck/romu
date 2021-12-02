use core::mem::{size_of, transmute};

use crate::{generate_seed, split_mix_64, SeedSource};

/// Implements `RomuTrio` with 512-bit width.
pub struct Rng512 {
    x: [u64; 8],
    y: [u64; 8],
    z: [u64; 8],
    seed_source: SeedSource,
}

impl Default for Rng512 {
    fn default() -> Self {
        let mut rng = Self {
            x: [0u64; 8],
            y: [0u64; 8],
            z: [0u64; 8],
            seed_source: SeedSource::Fixed,
        };
        rng.seed();
        rng
    }
}

impl Rng512 {
    /// Creates a new [`Rng512`] with a seed from the best available randomness source.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new [`Rng512`] from the given eight 64-bit seeds.
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

        Self {
            x: [
                lane0[0], lane1[0], lane2[0], lane3[0], lane4[0], lane5[0], lane6[0], lane7[0],
            ],
            y: [
                lane0[1], lane1[1], lane2[1], lane3[1], lane4[1], lane5[1], lane6[1], lane7[1],
            ],
            z: [
                lane0[2], lane1[2], lane2[2], lane3[2], lane4[2], lane5[2], lane6[2], lane7[2],
            ],
            seed_source: SeedSource::User,
        }
    }

    /// Creates a new [`Rng512`] from the given eight 192-bit seeds.
    ///
    /// The seeds should be from a high randomness source.
    ///
    /// If the seeds are of low quality, user should call [`Rng512::mix()`] to improve the quality of the
    /// first couple of random numbers.
    ///
    /// # Notice
    /// The variables must be seeded such that at least one bit of state is non-zero.
    pub const fn from_seed_with_192bit(seeds: [[u64; 3]; 8]) -> Self {
        Self {
            x: [
                seeds[0][0],
                seeds[1][0],
                seeds[2][0],
                seeds[3][0],
                seeds[4][0],
                seeds[5][0],
                seeds[6][0],
                seeds[7][0],
            ],
            y: [
                seeds[0][1],
                seeds[1][1],
                seeds[2][1],
                seeds[3][1],
                seeds[4][1],
                seeds[5][1],
                seeds[6][1],
                seeds[7][1],
            ],
            z: [
                seeds[0][2],
                seeds[1][2],
                seeds[2][2],
                seeds[3][2],
                seeds[4][2],
                seeds[5][2],
                seeds[6][2],
                seeds[7][2],
            ],
            seed_source: SeedSource::User,
        }
    }

    /// Mixes the states, which should improve the quality of the random numbers.
    ///
    /// Should be called when having (re-)seeded the generator with a fixed value of low randomness.
    pub fn mix(&mut self) {
        (0..10).into_iter().for_each(|_| {
            self.next();
        });
    }

    /// Re-seeds the [`Rng512`] from the best available randomness source.
    pub fn seed(&mut self) {
        let mut memory_address = self as *const _ as u64;
        let mut seed_source = SeedSource::Fixed;

        self.x
            .iter_mut()
            .zip(self.y.iter_mut())
            .zip(self.z.iter_mut())
            .for_each(|((x, y), z)| {
                let (lane, source) = generate_seed(memory_address);

                *x = lane[0];
                *y = lane[1];
                *z = lane[2];

                seed_source = source;
                memory_address = memory_address.wrapping_add(1);
            });

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

        self.x
            .iter_mut()
            .zip(self.y.iter_mut())
            .zip(self.z.iter_mut())
            .zip(xp.iter())
            .zip(yp.iter())
            .zip(zp.iter())
            .for_each(|(((((x, y), z), xp), yp), zp)| {
                *x = zp.wrapping_mul(0xD3833E804F4C574B);
                let ty = yp.wrapping_sub(*xp);
                *y = ty.wrapping_shl(12) | ty.wrapping_shr(52);
                let tz = zp.wrapping_sub(*yp);
                *z = tz.wrapping_shl(44) | tz.wrapping_shr(20);
            });

        xp
    }

    /// Generates eight random u64 values.
    #[inline(always)]
    pub fn u64x8(&mut self) -> [u64; 8] {
        self.next()
    }

    /// Fills a mutable `[u8]` slice with random values.
    pub fn fill_bytes(&mut self, slice: &mut [u8]) {
        const CHUNK_SIZE: usize = 8 * size_of::<u64>();
        assert!(size_of::<[u64; 8]>() == size_of::<[u8; CHUNK_SIZE]>());

        let mut chunks = slice.chunks_exact_mut(CHUNK_SIZE);
        for chunk in &mut chunks {
            let data: [u8; CHUNK_SIZE] = unsafe { transmute(self.next()) };
            chunk.copy_from_slice(&data)
        }

        let data: [u8; CHUNK_SIZE] = unsafe { transmute(self.next()) };
        chunks
            .into_remainder()
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = data[i]);
    }
}
