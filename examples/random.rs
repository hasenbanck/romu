#![cfg_attr(feature = "unstable_simd", feature(portable_simd))]
use std::io::{BufWriter, Write};

fn main() {
    #[cfg(not(feature = "unstable_simd"))]
    {
        use romu::Rng;

        let mut buffer = BufWriter::new(std::io::stdout());

        let rng = match Rng::new() {
            Ok(rng) => rng,
            Err(_) => Rng::from_seed_with_64bit(0x8D463F9844EDA4B8),
        };

        loop {
            buffer
                .write_all(&rng.u64().to_ne_bytes())
                .expect("can't write to stdout");
        }
    }
    #[cfg(feature = "unstable_simd")]
    {
        use core::simd::Simd;

        use romu::Rng512;

        let mut buffer = BufWriter::new(std::io::stdout());

        let rng = match Rng512::new() {
            Ok(rng) => rng,
            Err(_) => Rng512::from_seed_with_64bit([
                0x5700AC4B4733ACD9,
                0x90E78CD7468E6B3E,
                0x3BCA41CBE8271B61,
                0x4FF88BDB301DA413,
                0xE2A079B474C213CB,
                0x73FF716A694B317A,
                0xDED957D2AF617106,
                0x611516103678A29B,
            ]),
        };

        loop {
            let numbers = Simd::to_array(rng.u64x8());
            for number in numbers.iter() {
                buffer
                    .write_all(&number.to_ne_bytes())
                    .expect("can't write to stdout");
            }
        }
    }
}
