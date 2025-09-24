//! contig-core: contiguous layout trait, helpers, and common adapters.
//!
//! Key pieces:
//! - [`Contig`] describes how a value occupies a range inside an `&[F]` slice and
//!   exposes read/write views for that range.
//! - [`TakeCursor`] is a tiny helper for carving non-overlapping ranges while assembling
//!   a struct layout.
//! - Ready-made adapters for scalars, [`Vec3`], dynamic arrays (`Dyn<[T]>`), and (optionally)
//!   nalgebra vectors/matrices so common building blocks slot into a contiguous buffer without
//!   boilerplate.
//!
//! The `contig-derive` crate emits config/layout/view types that implement [`Contig`], letting
//! complex user-defined structs share the same zero-copy API as these primitives.

use core::{marker::PhantomData, ops::Range};

// ---------- Core error/result ----------

/// Errors that can occur while computing contiguous layouts.
#[derive(Debug)]
pub enum LayoutError {
    /// The computed footprint would overflow `usize`.
    Overflow,
    /// The configuration yielded an invalid size (usually reported with context).
    InvalidSize(&'static str),
}
/// Convenient result alias that uses [`LayoutError`] as the error type.
pub type Result<T> = core::result::Result<T, LayoutError>;

// ---------- Slice range cursor (linear, disjoint) ----------

/// A tiny "allocator" that carves disjoint ranges from a linear buffer.
pub struct TakeCursor {
    idx: usize,
}
impl TakeCursor {
    /// Create a new cursor starting at index `0`.
    pub fn new() -> Self {
        Self { idx: 0 }
    }
    /// Reserve the next `n` slots in the buffer and return their range.
    pub fn take_range(&mut self, n: usize) -> Range<usize> {
        let start = self.idx;
        self.idx = start
            .checked_add(n)
            .expect("overflow in TakeCursor::take_range");
        start..self.idx
    }
    /// Finish carving ranges and report the total footprint that was consumed.
    pub fn finish(self) -> usize {
        self.idx
    }
}

// ---------- Contig trait ----------

/// Describes how a value occupies and views a region inside a flat `[F]` slice.
///
/// Types implement this trait to participate in the zero-copy layout APIs
/// provided by `contig`.  Implementations describe how to size a value at
/// runtime, how many scalar slots are required, and how to materialize
/// read-only or mutable views into a backing slice.
pub trait Contig<F> {
    /// Runtime configuration required to size this value (e.g. lengths, nested configs).
    type Config;
    /// Fully computed layout metadata (cached by callers).
    type Layout;
    /// Read-only view type borrowing from the backing slice.
    type ConstView<'a>: 'a
    where
        F: 'a,
        Self::Layout: 'a;
    /// Mutable view type borrowing from the backing slice.
    type MutView<'a>: 'a
    where
        F: 'a,
        Self::Layout: 'a;

    /// Compute layout metadata from configuration, returning a fully-populated [`Self::Layout`].
    fn layout(config: &Self::Config) -> Result<Self::Layout>;
    /// Total scalar footprint required by a value with this layout.
    fn len(layout: &Self::Layout) -> usize;
    /// Build a read-only view into `buf` using this layout.
    fn view<'a>(layout: &'a Self::Layout, buf: &'a [F]) -> Self::ConstView<'a>;
    /// Build a mutable view into `buf` using this layout.
    fn view_mut<'a>(layout: &'a Self::Layout, buf: &'a mut [F]) -> Self::MutView<'a>;
}

// ---------- Scalars ----------

/// Trivial layout marker for scalar types where the layout metadata carries no
/// additional information.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScalarLayout;

macro_rules! impl_contig_scalar {
    ($($t:ty),* $(,)?) => {
        $(
            impl Contig<$t> for $t {
                type Config = ();
                type Layout = ScalarLayout;
                type ConstView<'a> = &'a $t;
                type MutView<'a> = &'a mut $t;

                fn layout(_config: &Self::Config) -> Result<Self::Layout> {
                    Ok(ScalarLayout)
                }

                fn len(_layout: &Self::Layout) -> usize {
                    1
                }

                fn view<'a>(_layout: &'a Self::Layout, buf: &'a [$t]) -> Self::ConstView<'a> {
                    debug_assert!(buf.len() >= 1);
                    &buf[0]
                }

                fn view_mut<'a>(_layout: &'a Self::Layout, buf: &'a mut [$t]) -> Self::MutView<'a> {
                    debug_assert!(buf.len() >= 1);
                    &mut buf[0]
                }
            }
        )*
    };
}

impl_contig_scalar!(f32, f64);

// ---------- Vec3 ----------

/// Marker type representing a fixed `[F; 3]` contiguous vector.
#[derive(Clone, Copy, Debug)]
pub struct Vec3<F>(PhantomData<F>);

/// Read-only view across three consecutive scalars laid out as `x`, `y`, `z`.
pub struct Vec3View<'a, F> {
    pub(crate) slice: &'a [F],
} // len = 3
/// Mutable view across three consecutive scalars laid out as `x`, `y`, `z`.
pub struct Vec3ViewMut<'a, F> {
    pub(crate) slice: &'a mut [F],
} // len = 3

impl<'a, F> Vec3View<'a, F> {
    #[inline]
    /// Access the `x` component of the vector.
    pub fn x(&self) -> &F {
        &self.slice[0]
    }
    #[inline]
    /// Access the `y` component of the vector.
    pub fn y(&self) -> &F {
        &self.slice[1]
    }
    #[inline]
    /// Access the `z` component of the vector.
    pub fn z(&self) -> &F {
        &self.slice[2]
    }
}
impl<'a, F> Vec3ViewMut<'a, F> {
    #[inline]
    /// Access the mutable `x` component of the vector.
    pub fn x(&mut self) -> &mut F {
        &mut self.slice[0]
    }
    #[inline]
    /// Access the mutable `y` component of the vector.
    pub fn y(&mut self) -> &mut F {
        &mut self.slice[1]
    }
    #[inline]
    /// Access the mutable `z` component of the vector.
    pub fn z(&mut self) -> &mut F {
        &mut self.slice[2]
    }
    #[inline]
    /// Set all components in one call (requires the scalar to be `Copy`).
    pub fn set(&mut self, x: F, y: F, z: F)
    where
        F: Copy,
    {
        self.slice[0] = x;
        self.slice[1] = y;
        self.slice[2] = z;
    }
}

/// Layout metadata marker for [`Vec3`]; it carries no additional information.
#[derive(Clone, Copy, Debug, Default)]
pub struct Vec3Layout;

impl<F> Contig<F> for Vec3<F> {
    type Config = ();
    type Layout = Vec3Layout;
    type ConstView<'a>
        = Vec3View<'a, F>
    where
        F: 'a;
    type MutView<'a>
        = Vec3ViewMut<'a, F>
    where
        F: 'a;

    fn layout(_config: &Self::Config) -> Result<Self::Layout> {
        Ok(Vec3Layout)
    }

    fn len(_layout: &Self::Layout) -> usize {
        3
    }

    fn view<'a>(_layout: &'a Self::Layout, buf: &'a [F]) -> Self::ConstView<'a> {
        debug_assert!(buf.len() >= 3);
        Vec3View { slice: &buf[..3] }
    }

    fn view_mut<'a>(_layout: &'a Self::Layout, buf: &'a mut [F]) -> Self::MutView<'a> {
        debug_assert!(buf.len() >= 3);
        Vec3ViewMut {
            slice: &mut buf[..3],
        }
    }
}

// ---------- Dyn<[T]> (dynamic arrays) ----------

/// Marker type representing a runtime-sized slice of contiguous `T` values.
pub struct Dyn<T: ?Sized>(PhantomData<T>);

/// Configuration for a runtime-sized contiguous array of `T`.
#[derive(Clone, Copy, Debug)]
pub struct DynArrayConfig<TCfg> {
    /// Number of `T` elements to expose through the view.
    pub len: usize,
    /// Nested configuration used to size a single element.
    pub elem: TCfg,
}

/// Fully computed layout information for a runtime-sized array of `T`.
#[derive(Clone, Debug)]
pub struct DynArrayLayout<TLayout> {
    /// Number of elements contained in this layout.
    pub len: usize,
    /// Cached element layout metadata (reused when producing views).
    pub elem_layout: TLayout,
    /// Scalar footprint of a single element.
    pub elem_len: usize,
}

/// Immutable view into a contiguous run of `count` elements of type `T`.
pub struct DynArrayConstView<'a, F, T>
where
    T: Contig<F>,
    T::Layout: Clone,
{
    base: &'a [F],
    count: usize,
    elem_layout: T::Layout,
    elem_len: usize,
    _marker: PhantomData<&'a ()>,
}

/// Mutable view into a contiguous run of `count` elements of type `T`.
pub struct DynArrayMutView<'a, F, T>
where
    T: Contig<F>,
    T::Layout: Clone,
{
    base: &'a mut [F],
    count: usize,
    elem_layout: T::Layout,
    elem_len: usize,
    _marker: PhantomData<&'a ()>,
}

impl<'a, F, T> DynArrayConstView<'a, F, T>
where
    T: Contig<F>,
    T::Layout: Clone,
{
    #[inline]
    /// Number of elements contained in this view.
    pub fn len(&self) -> usize {
        self.count
    }
    #[inline]
    /// Fetch a read-only view for element `i` (panics in debug if out of bounds).
    pub fn get(&self, i: usize) -> T::ConstView<'_> {
        debug_assert!(i < self.count);
        let start = i * self.elem_len;
        let end = start + self.elem_len;
        T::view(&self.elem_layout, &self.base[start..end])
    }
}

impl<'a, F, T> DynArrayMutView<'a, F, T>
where
    T: Contig<F>,
    T::Layout: Clone,
{
    #[inline]
    /// Number of elements contained in this view.
    pub fn len(&self) -> usize {
        self.count
    }
    #[inline]
    /// Fetch a mutable view for element `i` (panics in debug if out of bounds).
    pub fn get_mut(&mut self, i: usize) -> T::MutView<'_> {
        debug_assert!(i < self.count);
        let start = i * self.elem_len;
        let end = start + self.elem_len;
        T::view_mut(&self.elem_layout, &mut self.base[start..end])
    }
    #[inline]
    /// Fetch a read-only view for element `i` (panics in debug if out of bounds).
    pub fn get(&self, i: usize) -> T::ConstView<'_> {
        debug_assert!(i < self.count);
        let start = i * self.elem_len;
        let end = start + self.elem_len;
        T::view(&self.elem_layout, &self.base[start..end])
    }
}

// Dynamic array adapter backed by consecutive `T` layouts.
impl<F, T> Contig<F> for Dyn<[T]>
where
    T: Contig<F> + 'static,
    T::Layout: Clone + 'static,
{
    type Config = DynArrayConfig<T::Config>;
    type Layout = DynArrayLayout<T::Layout>;
    type ConstView<'a>
        = DynArrayConstView<'a, F, T>
    where
        F: 'a,
        T::Layout: Clone;
    type MutView<'a>
        = DynArrayMutView<'a, F, T>
    where
        F: 'a,
        T::Layout: Clone;

    fn layout(config: &Self::Config) -> Result<Self::Layout> {
        let elem_layout = T::layout(&config.elem)?;
        let elem_len = T::len(&elem_layout);
        Ok(DynArrayLayout {
            len: config.len,
            elem_layout,
            elem_len,
        })
    }

    fn len(layout: &Self::Layout) -> usize {
        layout.len * layout.elem_len
    }

    fn view<'a>(layout: &'a Self::Layout, buf: &'a [F]) -> Self::ConstView<'a> {
        debug_assert!(buf.len() >= Self::len(layout));
        DynArrayConstView {
            base: buf,
            count: layout.len,
            elem_layout: layout.elem_layout.clone(),
            elem_len: layout.elem_len,
            _marker: PhantomData,
        }
    }

    fn view_mut<'a>(layout: &'a Self::Layout, buf: &'a mut [F]) -> Self::MutView<'a> {
        debug_assert!(buf.len() >= Self::len(layout));
        DynArrayMutView {
            base: buf,
            count: layout.len,
            elem_layout: layout.elem_layout.clone(),
            elem_len: layout.elem_len,
            _marker: PhantomData,
        }
    }
}

// ---------- Optional nalgebra interop ----------

#[cfg(feature = "nalgebra")]
/// Types that adapt nalgebra vectors and matrices to the [`Contig`] trait.
pub mod na_types {
    use super::*;
    use nalgebra as na;

    /// Configuration for a dynamic-column vector view.
    #[derive(Clone, Copy, Debug)]
    pub struct DynVectorConfig {
        /// Total number of elements in the vector.
        pub len: usize,
    }
    /// Layout metadata for a dynamic-column vector view.
    #[derive(Clone, Copy, Debug)]
    pub struct DynVectorLayout {
        /// Total number of elements in the vector.
        pub len: usize,
    }

    /// Marker type that adapts `nalgebra::DVector` to [`Contig`].
    pub struct NaDVector<F>(PhantomData<F>);

    impl<F> Contig<F> for NaDVector<F>
    where
        F: na::Scalar,
    {
        type Config = DynVectorConfig;
        type Layout = DynVectorLayout;
        type ConstView<'a>
            = na::DVectorView<'a, F>
        where
            F: 'a;
        type MutView<'a>
            = na::DVectorViewMut<'a, F>
        where
            F: 'a;

        fn layout(config: &Self::Config) -> Result<Self::Layout> {
            Ok(DynVectorLayout { len: config.len })
        }

        fn len(layout: &Self::Layout) -> usize {
            layout.len
        }

        fn view<'a>(layout: &'a Self::Layout, buf: &'a [F]) -> Self::ConstView<'a> {
            debug_assert!(buf.len() >= layout.len);
            na::DVectorView::from_slice(buf, layout.len)
        }

        fn view_mut<'a>(layout: &'a Self::Layout, buf: &'a mut [F]) -> Self::MutView<'a> {
            debug_assert!(buf.len() >= layout.len);
            na::DVectorViewMut::from_slice(buf, layout.len)
        }
    }

    /// Configuration for a dynamic matrix view.
    #[derive(Clone, Copy, Debug)]
    pub struct DynMatrixConfig {
        /// Number of rows in the matrix.
        pub rows: usize,
        /// Number of columns in the matrix.
        pub cols: usize,
    }
    /// Layout metadata for a dynamic matrix view.
    #[derive(Clone, Copy, Debug)]
    pub struct DynMatrixLayout {
        /// Number of rows in the matrix.
        pub rows: usize,
        /// Number of columns in the matrix.
        pub cols: usize,
    }

    /// Marker type that adapts `nalgebra::DMatrix` to [`Contig`].
    pub struct NaDMatrix<F>(PhantomData<F>);

    impl<F> Contig<F> for NaDMatrix<F>
    where
        F: na::Scalar,
    {
        type Config = DynMatrixConfig;
        type Layout = DynMatrixLayout;
        type ConstView<'a>
            = na::DMatrixView<'a, F>
        where
            F: 'a;
        type MutView<'a>
            = na::DMatrixViewMut<'a, F>
        where
            F: 'a;

        fn layout(config: &Self::Config) -> Result<Self::Layout> {
            Ok(DynMatrixLayout {
                rows: config.rows,
                cols: config.cols,
            })
        }

        fn len(layout: &Self::Layout) -> usize {
            layout.rows * layout.cols
        }

        fn view<'a>(layout: &'a Self::Layout, buf: &'a [F]) -> Self::ConstView<'a> {
            debug_assert!(buf.len() >= Self::len(layout));
            na::DMatrixView::from_slice_generic(buf, na::Dyn(layout.rows), na::Dyn(layout.cols))
        }

        fn view_mut<'a>(layout: &'a Self::Layout, buf: &'a mut [F]) -> Self::MutView<'a> {
            debug_assert!(buf.len() >= Self::len(layout));
            na::DMatrixViewMut::from_slice_generic(buf, na::Dyn(layout.rows), na::Dyn(layout.cols))
        }
    }
}

// ---------- Prelude ----------

/// Convenience re-exports for building `contig`-based layouts.
pub mod prelude {
    #[cfg(feature = "nalgebra")]
    pub use super::na_types::*;
    pub use super::{
        Contig, Dyn, DynArrayConfig, DynArrayConstView, DynArrayLayout, DynArrayMutView,
        LayoutError, Result, TakeCursor, Vec3, Vec3View, Vec3ViewMut,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn take_cursor_allocates_disjoint_ranges() {
        let mut cursor = TakeCursor::new();
        let first = cursor.take_range(2);
        let second = cursor.take_range(5);
        assert_eq!(first, 0..2);
        assert_eq!(second, 2..7);
        assert_eq!(cursor.finish(), 7);
    }

    #[test]
    fn scalar_contig_roundtrip() {
        let layout = f64::layout(&()).unwrap();
        assert_eq!(f64::len(&layout), 1);
        let mut buf = [0.0f64; 1];
        {
            let value = f64::view_mut(&layout, &mut buf);
            *value = 42.0;
        }
        let value = f64::view(&layout, &buf);
        assert_eq!(*value, 42.0);
    }

    #[test]
    fn vec3_contig_views_share_buffer() {
        let layout = Vec3::<f64>::layout(&()).unwrap();
        assert_eq!(Vec3::<f64>::len(&layout), 3);
        let mut buf = [0.0f64; 3];
        {
            let mut view = Vec3::<f64>::view_mut(&layout, &mut buf);
            view.set(1.0, 2.0, 3.0);
        }
        let view = Vec3::<f64>::view(&layout, &buf);
        assert_eq!(*view.x(), 1.0);
        assert_eq!(*view.y(), 2.0);
        assert_eq!(*view.z(), 3.0);
    }

    #[test]
    fn dyn_array_of_scalars() {
        let cfg = DynArrayConfig { len: 4, elem: () };
        let layout = Dyn::<[f64]>::layout(&cfg).unwrap();
        assert_eq!(Dyn::<[f64]>::len(&layout), 4);
        let mut buf = vec![0.0f64; 4];

        {
            let mut view = Dyn::<[f64]>::view_mut(&layout, &mut buf);
            assert_eq!(view.len(), 4);
            for i in 0..view.len() {
                *view.get_mut(i) = i as f64 + 1.0;
            }
        }

        let view = Dyn::<[f64]>::view(&layout, &buf);
        for i in 0..view.len() {
            assert_eq!(*view.get(i), i as f64 + 1.0);
        }
    }

    #[test]
    fn dyn_array_of_vec3() {
        let cfg = DynArrayConfig { len: 2, elem: () };
        let layout = Dyn::<[Vec3<f64>]>::layout(&cfg).unwrap();
        assert_eq!(Dyn::<[Vec3<f64>]>::len(&layout), 6);
        let mut buf = vec![0.0f64; 6];

        {
            let mut view = Dyn::<[Vec3<f64>]>::view_mut(&layout, &mut buf);
            view.get_mut(0).set(1.0, 2.0, 3.0);
            view.get_mut(1).set(4.0, 5.0, 6.0);
        }

        let view = Dyn::<[Vec3<f64>]>::view(&layout, &buf);
        let v0 = view.get(0);
        assert_eq!(*v0.x(), 1.0);
        assert_eq!(*v0.y(), 2.0);
        assert_eq!(*v0.z(), 3.0);
        let v1 = view.get(1);
        assert_eq!(*v1.x(), 4.0);
        assert_eq!(*v1.y(), 5.0);
        assert_eq!(*v1.z(), 6.0);
    }

    #[test]
    fn dyn_array_zero_length_has_zero_footprint() {
        let cfg = DynArrayConfig { len: 0, elem: () };
        let layout = Dyn::<[f64]>::layout(&cfg).unwrap();
        assert_eq!(Dyn::<[f64]>::len(&layout), 0);
        let buf: [f64; 0] = [];
        let view = Dyn::<[f64]>::view(&layout, &buf);
        assert_eq!(view.len(), 0);
    }
}

#[cfg(feature = "nalgebra")]
#[test]
fn nalgebra_contig_vector_roundtrip() {
    use crate::na_types::{DynVectorConfig, NaDVector};

    let cfg = DynVectorConfig { len: 3 };
    let layout = NaDVector::<f64>::layout(&cfg).unwrap();
    assert_eq!(NaDVector::<f64>::len(&layout), 3);
    let mut buf = vec![0.0f64; 3];
    {
        let mut view = NaDVector::<f64>::view_mut(&layout, &mut buf);
        for i in 0..view.len() {
            view[i] = i as f64;
        }
    }
    let view = NaDVector::<f64>::view(&layout, &buf);
    for i in 0..view.len() {
        assert_eq!(view[i], i as f64);
    }
}
