//! Implements traits from the `rand` crate to provide compatibility with the broader ecosystem.

use crate::Rng;

impl rand_core::RngCore for Rng {
    fn next_u32(&mut self) -> u32 {
        self.u32()
    }

    fn next_u64(&mut self) -> u64 {
        self.u64()
    }

    fn fill_bytes(&mut self, dst: &mut [u8]) {
        Rng::fill_bytes(self, dst)
    }
}
