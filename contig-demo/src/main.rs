use contig_core::na_types as na;
use contig_core::prelude::*;
use contig_demo::Vec3;
use contig_derive::contig;

#[contig(scalar = f64)]
pub struct Link {
    mass: f64,
    pos: Vec3<f64>,
}

#[contig(scalar = f64)]
pub struct Robot {
    #[contig(len)]
    links: Dyn<[Link]>,
    #[contig(len, elem_shape)]
    mats: Dyn<[na::NaDMatrix<f64>]>,
}

fn main() {
    let _ = core::any::TypeId::of::<Robot>();
}
