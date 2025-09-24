use contig_derive::contig;

#[contig(scalar = f64)]
struct WrongAttr(f64);

fn main() {}
