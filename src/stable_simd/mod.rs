#[cfg(all(
any(target_arch = "x86", target_arch = "x86_64"),
target_feature = "avx2"
))]
mod avx2;

#[cfg(all(
any(target_arch = "x86", target_arch = "x86_64"),
not(target_feature = "avx2"),
target_feature = "sse2"
))]
mod sse2;

#[cfg(not(all(
any(target_arch = "x86", target_arch = "x86_64"),
any(target_feature = "sse2", target_feature = "avx2"),
)))]
mod fallback;

#[cfg(all(
any(target_arch = "x86", target_arch = "x86_64"),
target_feature = "avx2"
))]
pub use avx2::RngWide;

#[cfg(all(
any(target_arch = "x86", target_arch = "x86_64"),
not(target_feature = "avx2"),
target_feature = "sse2"
))]
pub use sse2::RngWide;

#[cfg(not(all(
any(target_arch = "x86", target_arch = "x86_64"),
any(target_feature = "sse2", target_feature = "avx2"),
)))]
pub use fallback::RngWide;
