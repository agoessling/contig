use contig_derive::contig;

#[contig(scalar = f64)]
struct MissingImport {
    value: f64,
}

fn main() {
    let cfg = MissingImportCfg { value: () };
    let layout = MissingImportLayout::from_config(&cfg).unwrap();
    let mut buf = vec![0.0; layout.len()];
    let _ = MissingImport::view_mut(&layout, &mut buf);
}
