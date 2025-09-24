#![cfg(feature = "nalgebra")]

use contig_core::na_types::{DynMatrixConfig, NaDMatrix};
use contig_core::prelude::*;

#[test]
fn nadmatrix_contig_roundtrip() {
    let cfg = DynMatrixConfig { rows: 2, cols: 3 };
    let layout = NaDMatrix::<f64>::layout(&cfg).unwrap();
    assert_eq!(NaDMatrix::<f64>::len(&layout), 6);

    let mut buf = vec![0.0f64; 6];
    {
        let mut view = NaDMatrix::<f64>::view_mut(&layout, &mut buf);
        for i in 0..2 {
            for j in 0..3 {
                view[(i, j)] = (i * 10 + j) as f64;
            }
        }
    }

    let view = NaDMatrix::<f64>::view(&layout, &buf);
    for i in 0..2 {
        for j in 0..3 {
            assert_eq!(view[(i, j)], (i * 10 + j) as f64);
        }
    }
}
