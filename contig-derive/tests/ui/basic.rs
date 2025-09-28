use contig_core::prelude::*;
use contig_derive::contig;

#[contig(scalar = f64)]
struct Sample {
    #[contig(len)]
    dyns: Dyn<[f64]>,
    scalar: f64,
}

fn main() {
    let cfg = SampleCfg {
        dyns: DynArrayConfig { len: 2, elem: () },
        scalar: (),
    };
    let layout = SampleLayout::from_config(&cfg);
    assert_eq!(<Sample as Contig<f64>>::len(&layout), 3);
}
