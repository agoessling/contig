use contig_core::prelude::*;
use contig_derive::contig;
use contig_demo::Vec3;

#[contig(scalar = f64)]
struct Link {
    mass: f64,
    pos: Vec3<f64>,
}

#[contig(scalar = f64)]
struct Robot {
    #[contig(len)]
    links: Dyn<[Link]>,
    #[contig(len)]
    scalars: Dyn<[f64]>,
}

#[contig(scalar = f64)]
struct Nested {
    #[contig(len)]
    rows: Dyn<[Dyn<[f64]>]>,
}

#[test]
fn robot_layout_roundtrip() {
    let cfg = RobotCfg {
        links: DynArrayConfig {
            len: 2,
            elem: LinkCfg {
                mass: (),
                pos: (),
            },
        },
        scalars: DynArrayConfig { len: 4, elem: () },
    };

    let layout = RobotLayout::from_config(&cfg).expect("layout");
    let expected_len = Dyn::<[Link]>::len(&Dyn::<[Link]>::layout(&cfg.links).unwrap())
        + Dyn::<[f64]>::len(&Dyn::<[f64]>::layout(&cfg.scalars).unwrap());
    assert_eq!(Robot::len(&layout), expected_len);

    let mut buf = vec![0.0f64; layout.len()];
    {
        let mut view = layout.view(&mut buf);

        {
            let mut links = view.links();
            let mut first = links.get_mut(0);
            *first.mass() = 10.0;
            first.pos().set(1.0, 2.0, 3.0);

            let mut second = links.get_mut(1);
            *second.mass() = 20.0;
            second.pos().set(4.0, 5.0, 6.0);
        }

        {
            let mut scalars = view.scalars();
            for i in 0..scalars.len() {
                *scalars.get_mut(i) = i as f64 + 0.5;
            }
        }
    }

    let cview = layout.cview(&buf);
    let links = cview.links();
    let first = links.get(0);
    assert_eq!(*first.mass(), 10.0);
    assert_eq!(*first.pos().x(), 1.0);
    assert_eq!(*first.pos().y(), 2.0);
    assert_eq!(*first.pos().z(), 3.0);

    let second = links.get(1);
    assert_eq!(*second.mass(), 20.0);
    assert_eq!(*second.pos().x(), 4.0);
    assert_eq!(*second.pos().y(), 5.0);
    assert_eq!(*second.pos().z(), 6.0);

    let scalars = cview.scalars();
    for i in 0..scalars.len() {
        assert_eq!(*scalars.get(i), i as f64 + 0.5);
    }
}

#[test]
fn nested_dynamic_array_roundtrip() {
    let cfg = NestedCfg {
        rows: DynArrayConfig {
            len: 2,
            elem: DynArrayConfig { len: 3, elem: () },
        },
    };

    let layout = NestedLayout::from_config(&cfg).expect("layout");
    assert_eq!(Nested::len(&layout), 2 * 3);
    let mut buf = vec![0.0f64; layout.len()];

    {
        let mut view = layout.view(&mut buf);
        let mut rows = view.rows();
        for row_idx in 0..rows.len() {
            let mut row = rows.get_mut(row_idx);
            for col_idx in 0..row.len() {
                *row.get_mut(col_idx) = (row_idx * 10 + col_idx) as f64;
            }
        }
    }

    let cview = layout.cview(&buf);
    let rows = cview.rows();
    for row_idx in 0..rows.len() {
        let row = rows.get(row_idx);
        for col_idx in 0..row.len() {
            assert_eq!(*row.get(col_idx), (row_idx * 10 + col_idx) as f64);
        }
    }
}
