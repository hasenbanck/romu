#![cfg_attr(feature = "unstable_simd", feature(portable_simd))]
use std::io::{BufWriter, Write};

fn main() {
    #[cfg(not(feature = "unstable_simd"))]
    {
        use romu::Rng;

        let mut buffer = BufWriter::new(std::io::stdout());

        let rng = Rng::new();

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

        let rng = Rng512::new();

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
