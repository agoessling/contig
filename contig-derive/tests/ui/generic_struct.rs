use contig_derive::contig;

#[contig(scalar = f64)]
struct Generic<T> {
    value: T,
}

fn main() {}
