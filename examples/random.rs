#![cfg_attr(feature = "unstable_simd", feature(portable_simd))]
use std::io::{BufWriter, Write};

fn main() {
    #[cfg(not(feature = "unstable_simd"))]
    {
        use romu::Rng;

        let mut buffer = BufWriter::new(std::io::stdout());

        let rng = match Rng::new() {
            Ok(rng) => rng,
            Err(_) => {
                Rng::from_seed_with_64bit(
                   0x8d463f9844eda4b8
                )
            }
        };

        loop {
            buffer.write_all(&rng.u64().to_ne_bytes()).expect("can't write to stdout");
        }
    }
    #[cfg(feature = "unstable_simd")]
    {
        use romu::Rng512;
        use core::simd::Simd;

        let mut buffer = BufWriter::new(std::io::stdout());

        let mut rng = match Rng512::new() {
            Ok(rng) => rng,
            Err(_) => {
                Rng512::from_seed_with_64bit([
                    0x5700ac4b4733acd9,
                    0x90e78cd7468e6b3e,
                    0x3bca41cbe8271b61,
                    0x4ff88bdb301da413,
                    0xe2a079b474c213cb,
                    0x73ff716a694b317a,
                    0xded957d2af617106,
                    0x611516103678a29b
                ])
            }
        };

        loop {
           let numbers = Simd::to_array(rng.u64x8());
            for number in numbers.iter() {
                buffer.write_all(&number.to_ne_bytes()).expect("can't write to stdout");
            }
        }
    }
}
