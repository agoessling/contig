use contig_derive::contig;
use contig_core::prelude::*;
use contig_core::na_types as na;

#[contig]
struct Link {
    mass: f64,
    pos:  Vec3<f64>,
}

#[contig]
struct Robot {
    #[contig(len)]
    links: Dyn<[Link]>,
    #[contig(len, elem_shape)]
    mats:  Dyn<[na::NaDMatrix<f64>]>,
}

fn main() {
    let cfg = RobotCfg {
        links: Robot_links_Cfg { len: 3 },
        mats: Robot_mats_Cfg {
            len: 2,
            elem: Robot_mats_elem_Cfg { shape: (6, 6) },
        },
    };

    let layout = RobotLayout::from_config(&cfg).unwrap();
    let mut buf = vec![0.0_f64; layout.len()];

    {
        let mut v = layout.view(&mut buf);
        *v.links().get_mut(0).mass() = 10.0;
        v.mats(); // contiguous block of matrices
    }

    println!("OK, buffer size = {}", layout.len());
}
