# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/)
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2025-08-21

### Added

- Added compatibility for the rand ecosystem by @MathisWellmann (#5)

## Changed

- Only provide one wide generator and provide a SSE2 version.

### Updated

- Raised minimal supported Rust version.
- Target getrandom v0.3

## [0.6.0] - 2024-03-23

### Updated

- Make unstable_simd compatible with the current nightly release. 

## [0.5.1] - 2020-12-03

### Added

- Add Default implementations for the wide Rng types.

## [0.5.0] - 2020-12-02

### Updated

- Provides special handwritten AVX2 versions using intrinsics on stable.

## [0.4.1] - 2020-12-01

### Updated

- Both stable and unstable SIMD generators generate the results.

## [0.4.0] - 2020-12-01

### Updated

- Remove inner mutability for SIMD types to support stable SIMD.

### Added

- Support stable SIMD using auto-vectorization.

## [0.3.0] - 2020-11-30

### Updated

- Rework the seed generation and make it more consistent.
- Make the source which was used for the seed generation queryable by the user.

## [0.2.1] - 2020-11-29

### Updated

- Always inline range_* function calls.

## [0.2.0] - 2020-11-29

### Updated

- Changed SIMD types to use inner mutability.

### Added

- mod_u* methods to Rng 
- mod_u* functions for the TLS. 

## [0.1.2] - 2021-11-28

### Added

- Initial release.
