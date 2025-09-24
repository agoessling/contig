use contig_core::prelude::*;
use contig_derive::contig;

#[contig(scalar = f64)]
struct Sample {
    #[contig(len)]
    dyns: Dyn<[f64]>,
    vec: Vec3<f64>,
}

fn main() {
    let cfg = SampleCfg {
        dyns: DynArrayConfig { len: 2, elem: () },
        vec: (),
    };
    let layout = SampleLayout::from_config(&cfg).unwrap();
    assert_eq!(<Sample as Contig<f64>>::len(&layout), 5);
}
