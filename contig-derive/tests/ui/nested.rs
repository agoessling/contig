use contig_core::prelude::*;
use contig_derive::contig;

#[contig(scalar = f64)]
struct Inner {
    value: f64,
}

#[contig(scalar = f64)]
struct Outer {
    #[contig(len)]
    inners: Dyn<[Inner]>,
}

fn main() {
    let cfg = OuterCfg {
        inners: DynArrayConfig {
            len: 2,
            elem: InnerCfg { value: () },
        },
    };
    let layout = OuterLayout::from_config(&cfg).unwrap();
    assert_eq!(Outer::len(&layout), 2);
}
