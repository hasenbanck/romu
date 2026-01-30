use core::{
    mem::{size_of, transmute},
    simd::{u64x8, Simd},
};

use crate::{generate_seed, split_mix_64, SeedSource};

/// Implements `RomuTrio` with 512-bit width.
pub struct RngWide {
    x: u64x8,
    y: u64x8,
    z: u64x8,
    seed_source: SeedSource,
}

impl Default for RngWide {
    fn default() -> Self {
        let mut rng = Self {
            x: Simd::from_array([0u64; 8]),
            y: Simd::from_array([0u64; 8]),
            z: Simd::from_array([0u64; 8]),
            seed_source: SeedSource::Fixed,
        };
        rng.seed();
        rng
    }
}

impl RngWide {
    /// Creates a new [`RngWide`] with a seed from the best available randomness source.
    pub fn new() -> Self {
        Self::default()
    }

    /// Re-seeds the [`RngWide`] from the best available randomness source.
    pub fn seed(&mut self) {
        let memory_address = self as *const _ as u64;
        let (lane0, _) = generate_seed(memory_address);
        let (lane1, _) = generate_seed(memory_address.wrapping_add(1));
        let (lane2, _) = generate_seed(memory_address.wrapping_add(2));
        let (lane3, _) = generate_seed(memory_address.wrapping_add(3));
        let (lane4, _) = generate_seed(memory_address.wrapping_add(4));
        let (lane5, _) = generate_seed(memory_address.wrapping_add(5));
        let (lane6, _) = generate_seed(memory_address.wrapping_add(6));
        let (lane7, seed_source) = generate_seed(memory_address.wrapping_add(7));

        self.x = Simd::from_array([
            lane0[0], lane1[0], lane2[0], lane3[0], lane4[0], lane5[0], lane6[0], lane7[0],
        ]);
        self.y = Simd::from_array([
            lane0[1], lane1[1], lane2[1], lane3[1], lane4[1], lane5[1], lane6[1], lane7[1],
        ]);
        self.z = Simd::from_array([
            lane0[2], lane1[2], lane2[2], lane3[2], lane4[2], lane5[2], lane6[2], lane7[2],
        ]);
        self.seed_source = seed_source;
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

        Self {
            x: Simd::from_array([
                lane0[0], lane1[0], lane2[0], lane3[0], lane4[0], lane5[0], lane6[0], lane7[0],
            ]),
            y: Simd::from_array([
                lane0[1], lane1[1], lane2[1], lane3[1], lane4[1], lane5[1], lane6[1], lane7[1],
            ]),
            z: Simd::from_array([
                lane0[2], lane1[2], lane2[2], lane3[2], lane4[2], lane5[2], lane6[2], lane7[2],
            ]),
            seed_source: SeedSource::User,
        }
    }

    /// Creates a new [`RngWide`] from the given four 192-bit seeds.
    ///
    /// The seeds should be from a high randomness source.
    ///
    /// If the seeds are of low quality, user should call [`RngWide::mix()`] to improve the quality of the
    /// first couple of random numbers.
    ///
    /// # Notice
    /// The variables must be seeded such that at least one bit of state is non-zero.
    pub const fn from_seed_with_192bit(seeds: [[u64; 3]; 8]) -> Self {
        Self {
            x: Simd::from_array([
                seeds[0][0],
                seeds[1][0],
                seeds[2][0],
                seeds[3][0],
                seeds[4][0],
                seeds[5][0],
                seeds[6][0],
                seeds[7][0],
            ]),
            y: Simd::from_array([
                seeds[0][1],
                seeds[1][1],
                seeds[2][1],
                seeds[3][1],
                seeds[4][1],
                seeds[5][1],
                seeds[6][1],
                seeds[7][1],
            ]),
            z: Simd::from_array([
                seeds[0][2],
                seeds[1][2],
                seeds[2][2],
                seeds[3][2],
                seeds[4][2],
                seeds[5][2],
                seeds[6][2],
                seeds[7][2],
            ]),
            seed_source: SeedSource::User,
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

    /// The actual wide `RomuTrio` algorithm.
    ///
    /// Great for general purpose work, including huge jobs.
    /// Est. capacity = 2^75 bytes. State size = 192 bits.
    ///
    /// Copyright 2020 Mark A. Overton
    /// Licensed under Apache-2.0.
    #[inline(always)]
    fn next(&mut self) -> u64x8 {
        let xp = self.x;
        let yp = self.y;
        let zp = self.z;

        self.x = zp * u64x8::splat(0xD3833E804F4C574B);
        let y = yp - xp;
        self.y = y << 12 | y >> 52;
        let z = zp - yp;
        self.z = z << 44 | z >> 20;

        xp
    }

    /// Generates eight random u64 values.
    #[inline(always)]
    pub fn u64x8(&mut self) -> u64x8 {
        self.next()
    }

    /// Fills a mutable `[u8]` slice with random values.
    #[allow(clippy::manual_bits)]
    pub fn fill_bytes(&mut self, slice: &mut [u8]) {
        const CHUNK_SIZE: usize = 8 * size_of::<u64>();

        let mut chunks = slice.chunks_exact_mut(CHUNK_SIZE);
        for chunk in &mut chunks {
            let data: [u8; CHUNK_SIZE] = unsafe { transmute(self.u64x8()) };
            chunk.copy_from_slice(&data)
        }

        debug_assert_eq!(size_of::<[u8; CHUNK_SIZE]>(), size_of::<[u64; 8]>());
        let data: [u8; CHUNK_SIZE] = unsafe { transmute(self.u64x8()) };
        chunks
            .into_remainder()
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = data[i]);
    }
}
