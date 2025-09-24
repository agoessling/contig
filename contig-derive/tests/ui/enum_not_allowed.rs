use contig_derive::contig;

#[contig(scalar = f64)]
enum NotAllowed {
    A,
    B,
}

fn main() {}
