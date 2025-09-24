//! contig-core: contiguous layout trait, helpers, and common adapters
//!
//! Key pieces:
//! - [`Contig`] trait (how a type maps to a contiguous `&[F]` slice)
//! - [`TakeCursor`] for building struct layouts from disjoint ranges
//! - Built-in adapters: scalars, [`Vec3`], dynamic arrays via [`Dyn<[T]>`],
//!   and (optionally) nalgebra vector/matrix wrappers
//!
//! The `contig-derive` macro generates per-struct config/layout/view types
//! that implement [`Contig`], allowing nested zero-copy views over a single
//! contiguous scalar buffer.

use core::{marker::PhantomData, ops::Range};

// ---------- Core error/result ----------

#[derive(Debug)]
pub enum LayoutError {
    Overflow,
    InvalidSize(&'static str),
}
pub type Result<T> = core::result::Result<T, LayoutError>;

// ---------- Slice range cursor (linear, disjoint) ----------

/// A tiny "allocator" that carves disjoint ranges from a linear buffer.
pub struct TakeCursor {
    idx: usize,
}
impl TakeCursor {
    pub fn new() -> Self {
        Self { idx: 0 }
    }
    pub fn take_range(&mut self, n: usize) -> Range<usize> {
        let start = self.idx;
        self.idx = start
            .checked_add(n)
            .expect("overflow in TakeCursor::take_range");
        start..self.idx
    }
    pub fn finish(self) -> usize {
        self.idx
    }
}

// ---------- Contig trait ----------

/// Describes how a value occupies and views a region inside a flat `[F]` slice.
pub trait Contig<F> {
    /// Runtime configuration required to size this value.
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

    /// Compute layout metadata from configuration.
    fn layout(config: &Self::Config) -> Result<Self::Layout>;
    /// Total scalar footprint for this layout.
    fn len(layout: &Self::Layout) -> usize;
    /// Build a read-only view into `buf` using this layout.
    fn view<'a>(layout: &'a Self::Layout, buf: &'a [F]) -> Self::ConstView<'a>;
    /// Build a mutable view into `buf` using this layout.
    fn view_mut<'a>(layout: &'a Self::Layout, buf: &'a mut [F]) -> Self::MutView<'a>;
}

// ---------- Scalars ----------

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

#[derive(Clone, Copy, Debug)]
pub struct Vec3<F>(PhantomData<F>);

pub struct Vec3View<'a, F> {
    pub(crate) slice: &'a [F],
} // len = 3
pub struct Vec3ViewMut<'a, F> {
    pub(crate) slice: &'a mut [F],
} // len = 3

impl<'a, F> Vec3View<'a, F> {
    #[inline]
    pub fn x(&self) -> &F {
        &self.slice[0]
    }
    #[inline]
    pub fn y(&self) -> &F {
        &self.slice[1]
    }
    #[inline]
    pub fn z(&self) -> &F {
        &self.slice[2]
    }
}
impl<'a, F> Vec3ViewMut<'a, F> {
    #[inline]
    pub fn x(&mut self) -> &mut F {
        &mut self.slice[0]
    }
    #[inline]
    pub fn y(&mut self) -> &mut F {
        &mut self.slice[1]
    }
    #[inline]
    pub fn z(&mut self) -> &mut F {
        &mut self.slice[2]
    }
    #[inline]
    pub fn set(&mut self, x: F, y: F, z: F)
    where
        F: Copy,
    {
        self.slice[0] = x;
        self.slice[1] = y;
        self.slice[2] = z;
    }
}

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

pub struct Dyn<T: ?Sized>(PhantomData<T>);

#[derive(Clone, Copy, Debug)]
pub struct DynArrayConfig<TCfg> {
    pub len: usize,
    pub elem: TCfg,
}

#[derive(Clone, Debug)]
pub struct DynArrayLayout<TLayout> {
    pub len: usize,
    pub elem_layout: TLayout,
    pub elem_len: usize,
}

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
    pub fn len(&self) -> usize {
        self.count
    }
    #[inline]
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
    pub fn len(&self) -> usize {
        self.count
    }
    #[inline]
    pub fn get_mut(&mut self, i: usize) -> T::MutView<'_> {
        debug_assert!(i < self.count);
        let start = i * self.elem_len;
        let end = start + self.elem_len;
        T::view_mut(&self.elem_layout, &mut self.base[start..end])
    }
    #[inline]
    pub fn get(&self, i: usize) -> T::ConstView<'_> {
        debug_assert!(i < self.count);
        let start = i * self.elem_len;
        let end = start + self.elem_len;
        T::view(&self.elem_layout, &self.base[start..end])
    }
}

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
pub mod na_types {
    use super::*;
    use nalgebra as na;

    #[derive(Clone, Copy, Debug)]
    pub struct DynVectorConfig {
        pub len: usize,
    }
    #[derive(Clone, Copy, Debug)]
    pub struct DynVectorLayout {
        pub len: usize,
    }

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

    #[derive(Clone, Copy, Debug)]
    pub struct DynMatrixConfig {
        pub rows: usize,
        pub cols: usize,
    }
    #[derive(Clone, Copy, Debug)]
    pub struct DynMatrixLayout {
        pub rows: usize,
        pub cols: usize,
    }

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

pub mod prelude {
    #[cfg(feature = "nalgebra")]
    pub use super::na_types::*;
    pub use super::{
        Contig, Dyn, DynArrayConfig, DynArrayConstView, DynArrayLayout, DynArrayMutView,
        LayoutError, Result, TakeCursor, Vec3, Vec3View, Vec3ViewMut,
    };
}
