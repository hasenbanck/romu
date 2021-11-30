use core::{
    cell::Cell,
    mem::transmute,
    simd::{u64x2, u64x4, u64x8, Simd},
};

use crate::{generate_seed, split_mix_64, SeedSource};

#[cfg_attr(docsrs, doc(cfg(feature = "unstable_simd")))]
/// Implements `RomuTrio` with 2 lane SIMD (128-bit).
pub struct Rng128 {
    x: Cell<u64x2>,
    y: Cell<u64x2>,
    z: Cell<u64x2>,
    seed_source: Cell<SeedSource>,
}

impl Rng128 {
    /// Creates a new [`Rng128`] with a seed from the best available randomness source.
    pub fn new() -> Self {
        let rng = Self {
            x: Cell::new(Simd::from_array([0u64; 2])),
            y: Cell::new(Simd::from_array([0u64; 2])),
            z: Cell::new(Simd::from_array([0u64; 2])),
            seed_source: Cell::new(SeedSource::Fixed),
        };
        rng.seed();

        rng
    }

    /// Re-seeds the [`Rng128`] from the best available randomness source.
    pub fn seed(&self) {
        let memory_address = self as *const _ as u64;
        let (lane0, _) = generate_seed(memory_address);
        let (lane1, seed_source) = generate_seed(memory_address.wrapping_add(1));

        self.x.set(Simd::from_array([lane0[0], lane1[0]]));
        self.y.set(Simd::from_array([lane0[1], lane1[1]]));
        self.z.set(Simd::from_array([lane0[2], lane1[2]]));
        self.seed_source.set(seed_source)
    }

    /// Creates a new [`Rng128`] from the given two 64-bit seeds.
    ///
    /// The seeds should be from a high randomness source.
    pub const fn from_seed_with_64bit(seeds: [u64; 4]) -> Self {
        let lane0 = split_mix_64(seeds[0]);
        let lane1 = split_mix_64(seeds[1]);

        Self {
            x: Cell::new(Simd::from_array([lane0[0], lane1[0]])),
            y: Cell::new(Simd::from_array([lane0[1], lane1[1]])),
            z: Cell::new(Simd::from_array([lane0[2], lane1[2]])),
            seed_source: Cell::new(SeedSource::User),
        }
    }

    /// Creates a new [`Rng128`] from the given two 192-bit seeds.
    ///
    /// The seeds should be from a high randomness source.
    ///
    /// If the seeds are of low quality, user should call [`Rng128::mix()`] to improve the quality of the
    /// first couple of random numbers.
    ///
    /// # Notice
    /// The variables must be seeded such that at least one bit of state is non-zero.
    pub const fn from_seed_with_192bit(seeds: [[u64; 3]; 2]) -> Self {
        Self {
            x: Cell::new(Simd::from_array([seeds[0][0], seeds[1][0]])),
            y: Cell::new(Simd::from_array([seeds[0][1], seeds[1][1]])),
            z: Cell::new(Simd::from_array([seeds[0][2], seeds[1][2]])),
            seed_source: Cell::new(SeedSource::User),
        }
    }

    /// Mixes the states, which should improve the quality of the random numbers.
    ///
    /// Should be called when having (re-)seeded the generator with a fixed value of low randomness.
    pub fn mix(&self) {
        (0..10).into_iter().for_each(|_| {
            self.next();
        });
    }

    /// The actual SIMD `RomuTrio` algorithm.
    ///
    /// Great for general purpose work, including huge jobs.
    /// Est. capacity = 2^75 bytes. State size = 192 bits.
    ///
    /// Copyright 2020 Mark A. Overton
    /// Licensed under Apache-2.0.
    #[inline(always)]
    fn next(&self) -> u64x2 {
        let xp = self.x.get();
        let yp = self.y.get();
        let zp = self.z.get();

        self.x.set(zp * 0xD3833E804F4C574B);
        let y = yp - xp;
        self.y.set(y << 12 | y >> 52);
        let z = zp - yp;
        self.z.set(z << 44 | z >> 20);

        xp
    }

    /// Generates a random [u64x2] value.
    #[inline(always)]
    pub fn u64x2(&self) -> u64x2 {
        self.next()
    }

    /// Fills a mutable `[u8]` slice with random values.
    pub fn fill_bytes(&self, slice: &mut [u8]) {
        assert_eq!(
            core::mem::size_of::<[u8; 16]>(),
            core::mem::size_of::<u64x2>()
        );

        let mut chunks = slice.chunks_exact_mut(16);
        for chunk in &mut chunks {
            // Safety: Should be safe since we asserted that they are of the same size.
            let data: [u8; 16] = unsafe { transmute(self.u64x2()) };
            chunk.copy_from_slice(&data)
        }

        // Safety: Should be safe since we asserted that they are of the same size.
        let data: [u8; 16] = unsafe { transmute(self.u64x2()) };
        chunks
            .into_remainder()
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = data[i]);
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "unstable_simd")))]
/// Implements `RomuTrio` with 4 lane SIMD (256-bit).
pub struct Rng256 {
    x: Cell<u64x4>,
    y: Cell<u64x4>,
    z: Cell<u64x4>,
    seed_source: Cell<SeedSource>,
}

impl Rng256 {
    /// Creates a new [`Rng256`] with a seed from the best available randomness source.
    pub fn new() -> Self {
        let rng = Self {
            x: Cell::new(Simd::from_array([0u64; 4])),
            y: Cell::new(Simd::from_array([0u64; 4])),
            z: Cell::new(Simd::from_array([0u64; 4])),
            seed_source: Cell::new(SeedSource::Fixed),
        };
        rng.seed();

        rng
    }

    /// Re-seeds the [`Rng256`] from the best available randomness source.
    ///
    /// Returns `False` if no seed with good randomness could be generated.
    pub fn seed(&self) {
        let memory_address = self as *const _ as u64;
        let (lane0, _) = generate_seed(memory_address);
        let (lane1, _) = generate_seed(memory_address.wrapping_add(1));
        let (lane2, _) = generate_seed(memory_address.wrapping_add(2));
        let (lane3, seed_source) = generate_seed(memory_address.wrapping_add(3));

        self.x
            .set(Simd::from_array([lane0[0], lane1[0], lane2[0], lane3[0]]));
        self.y
            .set(Simd::from_array([lane0[1], lane1[1], lane2[1], lane3[1]]));
        self.z
            .set(Simd::from_array([lane0[2], lane1[2], lane2[2], lane3[2]]));
        self.seed_source.set(seed_source)
    }

    /// Creates a new [`Rng256`] from the given four 64-bit seeds.
    ///
    /// The seeds should be from a high randomness source.
    pub const fn from_seed_with_64bit(seeds: [u64; 4]) -> Self {
        let lane0 = split_mix_64(seeds[0]);
        let lane1 = split_mix_64(seeds[1]);
        let lane2 = split_mix_64(seeds[2]);
        let lane3 = split_mix_64(seeds[3]);

        Self {
            x: Cell::new(Simd::from_array([lane0[0], lane1[0], lane2[0], lane3[0]])),
            y: Cell::new(Simd::from_array([lane0[1], lane1[1], lane2[1], lane3[1]])),
            z: Cell::new(Simd::from_array([lane0[2], lane1[2], lane2[2], lane3[2]])),
            seed_source: Cell::new(SeedSource::User),
        }
    }

    /// Creates a new [`Rng256`] from the given four 192-bit seeds.
    ///
    /// The seeds should be from a high randomness source.
    ///
    /// If the seeds are of low quality, user should call [`Rng256::mix()`] to improve the quality of the
    /// first couple of random numbers.
    ///
    /// # Notice
    /// The variables must be seeded such that at least one bit of state is non-zero.
    pub const fn from_seed_with_192bit(seeds: [[u64; 3]; 4]) -> Self {
        Self {
            x: Cell::new(Simd::from_array([
                seeds[0][0],
                seeds[1][0],
                seeds[2][0],
                seeds[3][0],
            ])),
            y: Cell::new(Simd::from_array([
                seeds[0][1],
                seeds[1][1],
                seeds[2][1],
                seeds[3][1],
            ])),
            z: Cell::new(Simd::from_array([
                seeds[0][2],
                seeds[1][2],
                seeds[2][2],
                seeds[3][2],
            ])),
            seed_source: Cell::new(SeedSource::User),
        }
    }

    /// Mixes the states, which should improve the quality of the random numbers.
    ///
    /// Should be called when having (re-)seeded the generator with a fixed value of low randomness.
    pub fn mix(&self) {
        (0..10).into_iter().for_each(|_| {
            self.next();
        });
    }

    /// The actual SIMD `RomuTrio` algorithm.
    ///
    /// Great for general purpose work, including huge jobs.
    /// Est. capacity = 2^75 bytes. State size = 192 bits.
    ///
    /// Copyright 2020 Mark A. Overton
    /// Licensed under Apache-2.0.
    #[inline(always)]
    fn next(&self) -> u64x4 {
        let xp = self.x.get();
        let yp = self.y.get();
        let zp = self.z.get();

        self.x.set(zp * 0xD3833E804F4C574B);
        let y = yp - xp;
        self.y.set(y << 12 | y >> 52);
        let z = zp - yp;
        self.z.set(z << 44 | z >> 20);

        xp
    }

    /// Generates a random [u64x4] value.
    #[inline(always)]
    pub fn u64x4(&self) -> u64x4 {
        self.next()
    }

    /// Fills a mutable `[u8]` slice with random values.
    pub fn fill_bytes(&self, slice: &mut [u8]) {
        assert_eq!(
            core::mem::size_of::<[u8; 32]>(),
            core::mem::size_of::<u64x4>()
        );

        let mut chunks = slice.chunks_exact_mut(32);
        for chunk in &mut chunks {
            // Safety: Should be safe since we asserted that they are of the same size.
            let data: [u8; 32] = unsafe { transmute(self.u64x4()) };
            chunk.copy_from_slice(&data)
        }

        // Safety: Should be safe since we asserted that they are of the same size.
        let data: [u8; 32] = unsafe { transmute(self.u64x4()) };
        chunks
            .into_remainder()
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = data[i]);
    }
}

#[cfg_attr(docsrs, doc(cfg(feature = "unstable_simd")))]
/// Implements `RomuTrio` with 8 lane SIMD (512-bit).
pub struct Rng512 {
    x: Cell<u64x8>,
    y: Cell<u64x8>,
    z: Cell<u64x8>,
    seed_source: Cell<SeedSource>,
}

impl Rng512 {
    /// Creates a new [`Rng512`] with a seed from the best available randomness source.
    pub fn new() -> Self {
        let rng = Self {
            x: Cell::new(Simd::from_array([0u64; 8])),
            y: Cell::new(Simd::from_array([0u64; 8])),
            z: Cell::new(Simd::from_array([0u64; 8])),
            seed_source: Cell::new(SeedSource::Fixed),
        };
        rng.seed();

        rng
    }

    /// Re-seeds the [`Rng512`] from the best available randomness source.
    pub fn seed(&self) {
        let memory_address = self as *const _ as u64;
        let (lane0, _) = generate_seed(memory_address);
        let (lane1, _) = generate_seed(memory_address.wrapping_add(1));
        let (lane2, _) = generate_seed(memory_address.wrapping_add(2));
        let (lane3, _) = generate_seed(memory_address.wrapping_add(3));
        let (lane4, _) = generate_seed(memory_address.wrapping_add(4));
        let (lane5, _) = generate_seed(memory_address.wrapping_add(5));
        let (lane6, _) = generate_seed(memory_address.wrapping_add(6));
        let (lane7, seed_source) = generate_seed(memory_address.wrapping_add(7));

        self.x.set(Simd::from_array([
            lane0[0], lane1[0], lane2[0], lane3[0], lane4[0], lane5[0], lane6[0], lane7[0],
        ]));
        self.y.set(Simd::from_array([
            lane0[1], lane1[1], lane2[1], lane3[1], lane4[1], lane5[1], lane6[1], lane7[1],
        ]));
        self.z.set(Simd::from_array([
            lane0[2], lane1[2], lane2[2], lane3[2], lane4[2], lane5[2], lane6[2], lane7[2],
        ]));
        self.seed_source.set(seed_source)
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
            x: Cell::new(Simd::from_array([
                lane0[0], lane1[0], lane2[0], lane3[0], lane4[0], lane5[0], lane6[0], lane7[0],
            ])),
            y: Cell::new(Simd::from_array([
                lane0[1], lane1[1], lane2[1], lane3[1], lane4[1], lane5[1], lane6[1], lane7[1],
            ])),
            z: Cell::new(Simd::from_array([
                lane0[2], lane1[2], lane2[2], lane3[2], lane4[2], lane5[2], lane6[2], lane7[2],
            ])),
            seed_source: Cell::new(SeedSource::User),
        }
    }

    /// Creates a new [`Rng512`] from the given four 192-bit seeds.
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
            x: Cell::new(Simd::from_array([
                seeds[0][0],
                seeds[1][0],
                seeds[2][0],
                seeds[3][0],
                seeds[4][0],
                seeds[5][0],
                seeds[6][0],
                seeds[7][0],
            ])),
            y: Cell::new(Simd::from_array([
                seeds[0][1],
                seeds[1][1],
                seeds[2][1],
                seeds[3][1],
                seeds[4][1],
                seeds[5][1],
                seeds[6][1],
                seeds[7][1],
            ])),
            z: Cell::new(Simd::from_array([
                seeds[0][2],
                seeds[1][2],
                seeds[2][2],
                seeds[3][2],
                seeds[4][2],
                seeds[5][2],
                seeds[6][2],
                seeds[7][2],
            ])),
            seed_source: Cell::new(SeedSource::User),
        }
    }

    /// Mixes the states, which should improve the quality of the random numbers.
    ///
    /// Should be called when having (re-)seeded the generator with a fixed value of low randomness.
    pub fn mix(&self) {
        (0..10).into_iter().for_each(|_| {
            self.next();
        });
    }

    /// The actual SIMD `RomuTrio` algorithm.
    ///
    /// Great for general purpose work, including huge jobs.
    /// Est. capacity = 2^75 bytes. State size = 192 bits.
    ///
    /// Copyright 2020 Mark A. Overton
    /// Licensed under Apache-2.0.
    #[inline(always)]
    fn next(&self) -> u64x8 {
        let xp = self.x.get();
        let yp = self.y.get();
        let zp = self.z.get();

        self.x.set(zp * 0xD3833E804F4C574B);
        let y = yp - xp;
        self.y.set(y << 12 | y >> 52);
        let z = zp - yp;
        self.z.set(z << 44 | z >> 20);

        xp
    }

    /// Generates a random [u64x8] value.
    #[inline(always)]
    pub fn u64x8(&self) -> u64x8 {
        self.next()
    }

    /// Fills a mutable `[u8]` slice with random values.
    pub fn fill_bytes(&self, slice: &mut [u8]) {
        assert_eq!(
            core::mem::size_of::<[u8; 64]>(),
            core::mem::size_of::<u64x8>()
        );

        // Safety: Should be safe since we asserted that they are of the same size.
        let mut chunks = slice.chunks_exact_mut(64);
        for chunk in &mut chunks {
            let data: [u8; 64] = unsafe { transmute(self.u64x8()) };
            chunk.copy_from_slice(&data)
        }

        // Safety: Should be safe since we asserted that they are of the same size.
        let data: [u8; 64] = unsafe { transmute(self.u64x8()) };
        chunks
            .into_remainder()
            .iter_mut()
            .enumerate()
            .for_each(|(i, x)| *x = data[i]);
    }
}
