//! contig-demo utilities.
//!
//! This crate now hosts example `Contig` adapters that higher-level tests and
//! examples can rely on without pulling them into `contig-core` itself.

pub mod vec3;

pub use vec3::{Vec3, Vec3Layout, Vec3View, Vec3ViewMut};
