use std::io::{BufWriter, Write};

fn main() {
    use romu::Rng;

    let mut buffer = BufWriter::new(std::io::stdout());

    let rng = Rng::new();

    loop {
        buffer
            .write_all(&rng.u64().to_ne_bytes())
            .expect("can't write to stdout");
    }
}
