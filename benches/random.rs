use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use romu::{Rng, RngWide};

pub fn scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar");

    let rng = Rng::new();

    group.bench_function(BenchmarkId::new("u8", ""), |b| {
        b.iter(|| {
            black_box(rng.u8());
        })
    });

    group.bench_function(BenchmarkId::new("u16", ""), |b| {
        b.iter(|| {
            black_box(rng.u16());
        })
    });

    group.bench_function(BenchmarkId::new("u32", ""), |b| {
        b.iter(|| {
            black_box(rng.u32());
        })
    });

    group.bench_function(BenchmarkId::new("u64", ""), |b| {
        b.iter(|| {
            black_box(rng.u64());
        })
    });

    group.bench_function(BenchmarkId::new("usize", ""), |b| {
        b.iter(|| {
            black_box(rng.usize());
        })
    });

    group.bench_function(BenchmarkId::new("i8", ""), |b| {
        b.iter(|| {
            black_box(rng.i8());
        })
    });

    group.bench_function(BenchmarkId::new("i16", ""), |b| {
        b.iter(|| {
            black_box(rng.i16());
        })
    });

    group.bench_function(BenchmarkId::new("i32", ""), |b| {
        b.iter(|| {
            black_box(rng.i32());
        })
    });

    group.bench_function(BenchmarkId::new("i64", ""), |b| {
        b.iter(|| {
            black_box(rng.i64());
        })
    });

    group.bench_function(BenchmarkId::new("isize", ""), |b| {
        b.iter(|| {
            black_box(rng.isize());
        })
    });

    group.bench_function(BenchmarkId::new("f32", ""), |b| {
        b.iter(|| {
            black_box(rng.f32());
        })
    });

    group.bench_function(BenchmarkId::new("f64", ""), |b| {
        b.iter(|| {
            black_box(rng.f64());
        })
    });

    group.bench_function(BenchmarkId::new("bool", ""), |b| {
        b.iter(|| {
            black_box(rng.bool());
        })
    });

    group.finish();
}

pub fn mod_u(c: &mut Criterion) {
    let mut group = c.benchmark_group("mod");

    let rng = Rng::new();

    group.bench_function(BenchmarkId::new("u8", ""), |b| {
        b.iter(|| {
            black_box(rng.mod_u8(u8::MAX - 1));
        })
    });

    group.bench_function(BenchmarkId::new("u16", ""), |b| {
        b.iter(|| {
            black_box(rng.mod_u16(u16::MAX - 1));
        })
    });

    group.bench_function(BenchmarkId::new("u32", ""), |b| {
        b.iter(|| {
            black_box(rng.mod_u32(u32::MAX - 1));
        })
    });

    group.bench_function(BenchmarkId::new("u64", ""), |b| {
        b.iter(|| {
            black_box(rng.mod_u64(u64::MAX - 1));
        })
    });

    group.bench_function(BenchmarkId::new("usize", ""), |b| {
        b.iter(|| {
            black_box(rng.mod_usize(usize::MAX - 1));
        })
    });

    group.finish();
}

pub fn range(c: &mut Criterion) {
    let mut group = c.benchmark_group("range");

    let rng = Rng::new();

    group.bench_function(BenchmarkId::new("u8", ""), |b| {
        b.iter(|| {
            black_box(rng.range_u8(..u8::MAX - 1));
        })
    });

    group.bench_function(BenchmarkId::new("u16", ""), |b| {
        b.iter(|| {
            black_box(rng.range_u16(..u16::MAX - 1));
        })
    });

    group.bench_function(BenchmarkId::new("u32", ""), |b| {
        b.iter(|| {
            black_box(rng.range_u32(..u32::MAX - 1));
        })
    });

    group.bench_function(BenchmarkId::new("u64", ""), |b| {
        b.iter(|| {
            black_box(rng.range_u64(..u64::MAX - 1));
        })
    });

    group.bench_function(BenchmarkId::new("usize", ""), |b| {
        b.iter(|| {
            black_box(rng.range_usize(..usize::MAX - 1));
        })
    });

    group.bench_function(BenchmarkId::new("i8", ""), |b| {
        b.iter(|| {
            black_box(rng.range_i8((i8::MIN + 1)..i8::MAX - 1));
        })
    });

    group.bench_function(BenchmarkId::new("i16", ""), |b| {
        b.iter(|| {
            black_box(rng.range_i16((i16::MIN + 1)..i16::MAX - 1));
        })
    });

    group.bench_function(BenchmarkId::new("i32", ""), |b| {
        b.iter(|| {
            black_box(rng.range_i32((i32::MIN + 1)..i32::MAX - 1));
        })
    });

    group.bench_function(BenchmarkId::new("i64", ""), |b| {
        b.iter(|| {
            black_box(rng.range_i64((i64::MIN + 1)..i64::MAX - 1));
        })
    });

    group.bench_function(BenchmarkId::new("isize", ""), |b| {
        b.iter(|| {
            black_box(rng.range_isize((isize::MIN + 1)..isize::MAX - 1));
        })
    });

    group.finish();
}

pub fn bytes(c: &mut Criterion) {
    let mut group = c.benchmark_group("bytes");

    let rng = Rng::new();

    let size = 1024 * 1024; // 1 MiB
    group.throughput(Throughput::Bytes(size as u64));

    let mut buffer = vec![0u8; size];

    group.bench_with_input(BenchmarkId::new("Rng", size), &size, |b, &_s| {
        b.iter(|| rng.fill_bytes(&mut buffer));
    });
    black_box(buffer);

    let mut rng = RngWide::new();

    let mut buffer = vec![0u8; size];

    group.bench_with_input(BenchmarkId::new("RngWide", size), &size, |b, &_s| {
        b.iter(|| rng.fill_bytes(&mut buffer));
    });
    black_box(buffer);

    group.finish();
}

#[cfg(feature = "tls")]
pub fn tls(c: &mut Criterion) {
    let mut group = c.benchmark_group("tls");

    let rng = Rng::new();
    romu::seed();

    group.bench_function(BenchmarkId::new("u64", "instanced"), |b| {
        b.iter(|| {
            black_box(rng.u32());
        })
    });

    group.bench_function(BenchmarkId::new("u64", "thread local"), |b| {
        b.iter(|| {
            black_box(romu::u32());
        })
    });

    let rng = Rng::new();
    romu::seed();

    let count = 1024 * 1024;
    group.throughput(Throughput::Bytes(count));

    let mut buffer = vec![0u8; count as usize];
    group.bench_function(BenchmarkId::new("bytes", "instanced"), |b| {
        b.iter(|| {
            rng.fill_bytes(&mut buffer);
        })
    });
    black_box(buffer);

    let mut buffer = vec![0u8; count as usize];
    group.bench_function(BenchmarkId::new("bytes", "thread local"), |b| {
        b.iter(|| {
            romu::fill_bytes(&mut buffer);
        })
    });
    black_box(buffer);

    group.finish();
}

#[cfg(feature = "tls")]
criterion_group!(benches, scalar, mod_u, range, bytes, tls);
#[cfg(not(feature = "tls"))]
criterion_group!(benches, scalar, mod_u, range, bytes);

criterion_main!(benches);
