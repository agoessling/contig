//! contig-core: traits, helpers, and allowed-type wrappers
//!
//! This crate defines:
//! - Spec / DynSpec traits (how a type maps to a contiguous [F] slice)
//! - DynArgs (runtime dims), TakeCursor (linear range allocator)
//! - A few "allowed types":
//!     * Scalar F (f32 / f64)
//!     * Vec3<F> small fixed vector, with zero-copy views
//!     * Fixed array [T; N]     (static size)
//!     * Dyn<[T]>               (dynamic arrays with homogeneous elements)
//!     * (feature "nalgebra"): NaSVector/NaSMatrix (static), NaDVector/NaDMatrix (dynamic)
//!
//! The proc-macro in `contig-derive` will generate Layout + View types and
//! rely on these trait impls to produce field sub-views.

use core::ops::Range;

// ---------- Core error/result ----------

#[derive(Debug)]
pub enum LayoutError {
    Overflow,
    InvalidSize(&'static str),
}
pub type Result<T> = core::result::Result<T, LayoutError>;

// ---------- Runtime sizing arguments ----------

/// Runtime shape arguments for a particular field or element.
/// Only one of {len} or {shape} is typically set for a given node.
#[derive(Clone, Copy, Debug, Default)]
pub struct DynArgs {
    pub len: Option<usize>, // dynamic vector length / array element count
    pub shape: Option<(usize, usize)>, // dynamic matrix shape (rows, cols)
}
impl DynArgs {
    pub fn with_len(len: usize) -> Self {
        Self {
            len: Some(len),
            shape: None,
        }
    }
    pub fn with_shape(rows: usize, cols: usize) -> Self {
        Self {
            len: None,
            shape: Some((rows, cols)),
        }
    }
}

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

// ---------- Traits: Spec (static) / DynSpec (runtime) ----------

/// A statically-shaped value that occupies exactly STATIC_LEN scalars in a flat buffer.
pub trait Spec<F> {
    const STATIC_LEN: usize;

    type CView<'a>: 'a
    where
        F: 'a,
        Self: 'a;
    type MView<'a>: 'a
    where
        F: 'a,
        Self: 'a;

    /// Build a read-only view from an exact sub-slice (length == STATIC_LEN).
    fn cview<'a>(buf: &'a [F]) -> Self::CView<'a>;
    /// Build a mutable view from an exact sub-slice (length == STATIC_LEN).
    fn mview<'a>(buf: &'a mut [F]) -> Self::MView<'a>;
}

/// A dynamically-shaped value (vector, matrix, or array of T) whose size is
/// determined at configuration time and remains constant at runtime.
pub trait DynSpec<F> {
    type CView<'a>: 'a
    where
        F: 'a,
        Self: 'a;
    type MView<'a>: 'a
    where
        F: 'a,
        Self: 'a;

    /// Footprint (number of scalars) given runtime args for THIS node.
    fn dyn_len(args: &DynArgs) -> usize;

    /// Optional: when this node is an array-of-dynamic elements, you may also
    /// receive element args (e.g., per-matrix shape). Default ignores element args.
    fn dyn_len_with_elem(args: &DynArgs, _elem_args: &DynArgs) -> usize {
        Self::dyn_len(args)
    }

    /// Build a read-only view for THIS node, using its args.
    fn cview<'a>(buf: &'a [F], args: &DynArgs) -> Self::CView<'a>;
    /// Build a mutable view for THIS node, using its args.
    fn mview<'a>(buf: &'a mut [F], args: &DynArgs) -> Self::MView<'a>;

    /// Build a view for an "array of dynamic elements" node, where the element
    /// has its own DynArgs. Default delegates to cview/mview (no elements).
    fn cview_full<'a>(
        buf: &'a [F],
        args: &DynArgs,
        _elem_args: &DynArgs,
    ) -> Self::CView<'a> {
        Self::cview(buf, args)
    }
    fn mview_full<'a>(
        buf: &'a mut [F],
        args: &DynArgs,
        _elem_args: &DynArgs,
    ) -> Self::MView<'a> {
        Self::mview(buf, args)
    }
}

// ---------- Allowed type: Scalar F (f32/f64) ----------

macro_rules! impl_spec_scalar {
    ($t:ty) => {
        impl Spec<$t> for $t {
            const STATIC_LEN: usize = 1;
            type CView<'a>
                = &'a $t
            where
                $t: 'a;
            type MView<'a>
                = &'a mut $t
            where
                $t: 'a;

            #[inline]
            fn cview<'a>(buf: &'a [$t]) -> Self::CView<'a> {
                debug_assert_eq!(buf.len(), 1);
                &buf[0]
            }
            #[inline]
            fn mview<'a>(buf: &'a mut [$t]) -> Self::MView<'a> {
                debug_assert_eq!(buf.len(), 1);
                &mut buf[0]
            }
        }

        impl DynSpec<$t> for $t {
            type CView<'a>
                = &'a $t
            where
                $t: 'a;
            type MView<'a>
                = &'a mut $t
            where
                $t: 'a;

            #[inline]
            fn dyn_len(_args: &DynArgs) -> usize {
                1
            }

            #[inline]
            fn cview<'a>(buf: &'a [$t], _args: &DynArgs) -> Self::CView<'a> {
                < $t as Spec<$t> >::cview(buf)
            }

            #[inline]
            fn mview<'a>(buf: &'a mut [$t], _args: &DynArgs) -> Self::MView<'a> {
                < $t as Spec<$t> >::mview(buf)
            }
        }
    };
}
impl_spec_scalar!(f32);
impl_spec_scalar!(f64);

// ---------- Allowed type: Vec3<F> (small fixed vector) ----------

#[derive(Clone, Copy, Debug)]
pub struct Vec3<F>(core::marker::PhantomData<F>);

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

impl<F> Spec<F> for Vec3<F> {
    const STATIC_LEN: usize = 3;
    type CView<'a>
        = Vec3View<'a, F>
    where
        F: 'a;
    type MView<'a>
        = Vec3ViewMut<'a, F>
    where
        F: 'a;

    #[inline]
    fn cview<'a>(buf: &'a [F]) -> Self::CView<'a> {
        debug_assert_eq!(buf.len(), 3);
        Vec3View { slice: buf }
    }
    #[inline]
    fn mview<'a>(buf: &'a mut [F]) -> Self::MView<'a> {
        debug_assert_eq!(buf.len(), 3);
        Vec3ViewMut { slice: buf }
    }
}

impl<F> DynSpec<F> for Vec3<F> {
    type CView<'a>
        = Vec3View<'a, F>
    where
        F: 'a,
        Vec3<F>: 'a;
    type MView<'a>
        = Vec3ViewMut<'a, F>
    where
        F: 'a,
        Vec3<F>: 'a;

    #[inline]
    fn dyn_len(_args: &DynArgs) -> usize {
        Vec3::<F>::STATIC_LEN
    }

    #[inline]
    fn cview<'a>(buf: &'a [F], _args: &DynArgs) -> Self::CView<'a> {
        <Vec3<F> as Spec<F>>::cview(buf)
    }

    #[inline]
    fn mview<'a>(buf: &'a mut [F], _args: &DynArgs) -> Self::MView<'a> {
        <Vec3<F> as Spec<F>>::mview(buf)
    }
}

// ---------- Allowed type: Fixed array [T; N] (static) ----------

impl<F, T, const N: usize> Spec<F> for [T; N]
where
    T: Spec<F>,
{
    const STATIC_LEN: usize = N * T::STATIC_LEN;

    type CView<'a>
        = ArrayCView<'a, F, T, N>
    where
        F: 'a,
        T: 'a;
    type MView<'a>
        = ArrayMView<'a, F, T, N>
    where
        F: 'a,
        T: 'a;

    fn cview<'a>(buf: &'a [F]) -> Self::CView<'a> {
        debug_assert_eq!(buf.len(), Self::STATIC_LEN);
        ArrayCView {
            base: buf,
            _marker: core::marker::PhantomData,
        }
    }
    fn mview<'a>(buf: &'a mut [F]) -> Self::MView<'a> {
        debug_assert_eq!(buf.len(), Self::STATIC_LEN);
        ArrayMView {
            base: buf,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<F, T, const N: usize> DynSpec<F> for [T; N]
where
    T: Spec<F>,
{
    type CView<'a>
        = ArrayCView<'a, F, T, N>
    where
        F: 'a,
        [T; N]: 'a;
    type MView<'a>
        = ArrayMView<'a, F, T, N>
    where
        F: 'a,
        [T; N]: 'a;

    #[inline]
    fn dyn_len(_args: &DynArgs) -> usize {
        <[T; N] as Spec<F>>::STATIC_LEN
    }

    #[inline]
    fn cview<'a>(buf: &'a [F], _args: &DynArgs) -> Self::CView<'a> {
        <[T; N] as Spec<F>>::cview(buf)
    }

    #[inline]
    fn mview<'a>(buf: &'a mut [F], _args: &DynArgs) -> Self::MView<'a> {
        <[T; N] as Spec<F>>::mview(buf)
    }
}

/// Read-only view over a fixed array of `T` packed consecutively.
pub struct ArrayCView<'a, F, T: Spec<F>, const N: usize> {
    base: &'a [F],
    _marker: core::marker::PhantomData<&'a T>,
}
/// Mutable view over a fixed array of `T` packed consecutively.
pub struct ArrayMView<'a, F, T: Spec<F>, const N: usize> {
    base: &'a mut [F],
    _marker: core::marker::PhantomData<&'a T>,
}

impl<'a, F, T: Spec<F>, const N: usize> ArrayCView<'a, F, T, N> {
    #[inline]
    pub fn len(&self) -> usize {
        N
    }
    #[inline]
    pub fn get(&self, i: usize) -> T::CView<'_> {
        let span = T::STATIC_LEN;
        let start = i * span;
        T::cview(&self.base[start..start + span])
    }
}
impl<'a, F, T: Spec<F>, const N: usize> ArrayMView<'a, F, T, N> {
    #[inline]
    pub fn len(&self) -> usize {
        N
    }
    #[inline]
    pub fn get_mut(&mut self, i: usize) -> T::MView<'_> {
        let span = T::STATIC_LEN;
        let start = i * span;
        T::mview(&mut self.base[start..start + span])
    }
}

// ---------- Allowed type: Dyn<[T]> (dynamic array, homogeneous elements) ----------

/// A dynamic array container (homogeneous) recognized by the macro.
/// The element `T` may be `Spec<F>` (static) or itself a dynamic type.
pub struct Dyn<T: ?Sized>(core::marker::PhantomData<T>);

/// For Dyn<[T]>, the node's args carry `len = N`.
/// Element args (if any) are passed via the extra elem_args parameter.
impl<F, T> DynSpec<F> for Dyn<[T]>
where
    T: DynSpec<F>,
{
    type CView<'a>
        = DynArrayCView<'a, F, T>
    where
        F: 'a,
        T: 'a;
    type MView<'a>
        = DynArrayMView<'a, F, T>
    where
        F: 'a,
        T: 'a;

    fn dyn_len(args: &DynArgs) -> usize {
        let n = args.len.expect("Dyn<[T]> requires len");
        if n == 0 {
            return 0;
        }
        n * T::dyn_len(&DynArgs::default())
    }

    fn dyn_len_with_elem(args: &DynArgs, elem_args: &DynArgs) -> usize {
        let n = args.len.expect("Dyn<[T]> requires len");
        if n == 0 {
            return 0;
        }
        n * T::dyn_len(elem_args)
    }

    fn cview<'a>(buf: &'a [F], args: &DynArgs) -> Self::CView<'a> {
        let count = args.len.expect("Dyn<[T]> requires len");
        let elem_span = if count == 0 {
            0
        } else {
            debug_assert_eq!(buf.len() % count, 0);
            buf.len() / count
        };
        DynArrayCView {
            base: buf,
            count,
            elem_args: DynArgs::default(),
            elem_span,
            _marker: core::marker::PhantomData,
        }
    }

    fn mview<'a>(buf: &'a mut [F], args: &DynArgs) -> Self::MView<'a> {
        let count = args.len.expect("Dyn<[T]> requires len");
        let elem_span = if count == 0 {
            0
        } else {
            debug_assert_eq!(buf.len() % count, 0);
            buf.len() / count
        };
        DynArrayMView {
            base: buf,
            count,
            elem_args: DynArgs::default(),
            elem_span,
            _marker: core::marker::PhantomData,
        }
    }

    fn cview_full<'a>(buf: &'a [F], args: &DynArgs, elem_args: &DynArgs) -> Self::CView<'a> {
        let count = args.len.expect("Dyn<[T]> requires len");
        let elem_span = if count == 0 {
            0
        } else {
            let span = T::dyn_len(elem_args);
            debug_assert_eq!(buf.len(), count * span);
            span
        };
        DynArrayCView {
            base: buf,
            count,
            elem_args: *elem_args,
            elem_span,
            _marker: core::marker::PhantomData,
        }
    }

    fn mview_full<'a>(
        buf: &'a mut [F],
        args: &DynArgs,
        elem_args: &DynArgs,
    ) -> Self::MView<'a> {
        let count = args.len.expect("Dyn<[T]> requires len");
        let elem_span = if count == 0 {
            0
        } else {
            let span = T::dyn_len(elem_args);
            debug_assert_eq!(buf.len(), count * span);
            span
        };
        DynArrayMView {
            base: buf,
            count,
            elem_args: *elem_args,
            elem_span,
            _marker: core::marker::PhantomData,
        }
    }
}

/// Read-only view over a Dyn<[T]> where T: DynSpec<F>
pub struct DynArrayCView<'a, F, T: DynSpec<F>> {
    base: &'a [F],
    count: usize,
    elem_args: DynArgs,
    elem_span: usize,
    _marker: core::marker::PhantomData<&'a T>,
}
/// Mutable view over a Dyn<[T]> where T: DynSpec<F>
pub struct DynArrayMView<'a, F, T: DynSpec<F>> {
    base: &'a mut [F],
    count: usize,
    elem_args: DynArgs,
    elem_span: usize,
    _marker: core::marker::PhantomData<&'a T>,
}

impl<'a, F, T: DynSpec<F>> DynArrayCView<'a, F, T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }
    #[inline]
    pub fn get(&self, i: usize) -> T::CView<'_> {
        debug_assert!(i < self.count);
        let span = self.elem_span;
        let start = i * span;
        T::cview(&self.base[start..start + span], &self.elem_args)
    }
}
impl<'a, F, T: DynSpec<F>> DynArrayMView<'a, F, T> {
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }
    #[inline]
    pub fn get_mut(&mut self, i: usize) -> T::MView<'_> {
        debug_assert!(i < self.count);
        let span = self.elem_span;
        let start = i * span;
        T::mview(&mut self.base[start..start + span], &self.elem_args)
    }
}

// ---------- (feature) nalgebra interop ----------

#[cfg(feature = "nalgebra")]
pub mod na_types {
    use super::*;
    use nalgebra as na;

    // Fixed-size nalgebra adapters (static)
    pub struct NaSVector<F, const N: usize>(core::marker::PhantomData<(F, [(); N])>);
    pub struct NaSMatrix<F, const R: usize, const C: usize>(
        core::marker::PhantomData<(F, [(); R], [(); C])>,
    );

    // Dynamic-size nalgebra adapters (dynamic)
    pub struct NaDVector<F>(core::marker::PhantomData<F>);
    pub struct NaDMatrix<F>(core::marker::PhantomData<F>);

    // Views (re-export nalgebra slice types for call sites)
    pub type DVecSlice<'a, F> = na::DVectorView<'a, F>;
    pub type DVecSliceMut<'a, F> = na::DVectorViewMut<'a, F>;
    pub type DMatSlice<'a, F> = na::DMatrixView<'a, F>;
    pub type DMatSliceMut<'a, F> = na::DMatrixViewMut<'a, F>;

    // --- Static vector/matrix: Spec ---

    impl<F, const N: usize> Spec<F> for NaSVector<F, N>
    where
        F: na::Scalar + Copy,
    {
        const STATIC_LEN: usize = N;
        type CView<'a>
            = na::SVector<F, N>
        where
            F: 'a;
        type MView<'a>
            = na::SVector<F, N>
        where
            F: 'a;

        fn cview(buf: &[F]) -> Self::CView<'_> {
            na::SVector::<F, N>::from_column_slice(buf)
        }
        fn mview(buf: &mut [F]) -> Self::MView<'_> {
            // For SVector, nalgebra expects owned data; however, constructing
            // from a slice copies. If you want zero-copy, prefer returning a
            // MatrixSlice view. To stay truly zero-copy, you can instead expose:
            //   type CView = na::Matrix<F, Const<N>, Const<1>, SliceStorage<'a, F, Const<N>, Const<1>, Const<1>, Const<N>>>
            // For brevity, we use the simple owned form here. In your real code,
            // replace with SliceStorage-based view types.
            na::SVector::<F, N>::from_column_slice(buf)
        }
    }

    impl<F, const R: usize, const C: usize> Spec<F> for NaSMatrix<F, R, C>
    where
        F: na::Scalar + Copy,
    {
        const STATIC_LEN: usize = R * C;
        type CView<'a>
            = na::DMatrix<F>
        where
            F: 'a;
        type MView<'a>
            = na::DMatrix<F>
        where
            F: 'a;

        fn cview(buf: &[F]) -> Self::CView<'_> {
            na::DMatrix::<F>::from_column_slice(R, C, buf)
        }
        fn mview(buf: &mut [F]) -> Self::MView<'_> {
            na::DMatrix::<F>::from_column_slice(R, C, buf)
        }
    }

    // --- Dynamic vector: DynSpec(len) ---

    impl<F> DynSpec<F> for NaDVector<F>
    where
        F: na::Scalar,
    {
        type CView<'a>
            = DVecSlice<'a, F>
        where
            F: 'a;
        type MView<'a>
            = DVecSliceMut<'a, F>
        where
            F: 'a;

        fn dyn_len(args: &DynArgs) -> usize {
            args.len.expect("NaDVector requires len")
        }
        fn cview<'a>(buf: &'a [F], args: &DynArgs) -> Self::CView<'a> {
            let n = args.len.expect("NaDVector requires len");
            na::DVectorView::from_slice(buf, n)
        }
        fn mview<'a>(buf: &'a mut [F], args: &DynArgs) -> Self::MView<'a> {
            let n = args.len.expect("NaDVector requires len");
            na::DVectorViewMut::from_slice(buf, n)
        }
    }

    // --- Dynamic matrix: DynSpec(shape) ---

    impl<F> DynSpec<F> for NaDMatrix<F>
    where
        F: na::Scalar,
    {
        type CView<'a>
            = DMatSlice<'a, F>
        where
            F: 'a;
        type MView<'a>
            = DMatSliceMut<'a, F>
        where
            F: 'a;

        fn dyn_len(args: &DynArgs) -> usize {
            let (r, c) = args.shape.expect("NaDMatrix requires shape");
            r * c
        }
        fn cview<'a>(buf: &'a [F], args: &DynArgs) -> Self::CView<'a> {
            let (r, c) = args.shape.expect("NaDMatrix requires shape");
            na::DMatrixView::from_slice_generic(buf, na::Dyn(r), na::Dyn(c))
        }
        fn mview<'a>(buf: &'a mut [F], args: &DynArgs) -> Self::MView<'a> {
            let (r, c) = args.shape.expect("NaDMatrix requires shape");
            na::DMatrixViewMut::from_slice_generic(buf, na::Dyn(r), na::Dyn(c))
        }
    }
}

// Re-export commonly used items for convenience in downstream code.
pub mod prelude {
    #[cfg(feature = "nalgebra")]
    pub use super::na_types::*;
    pub use super::{
        Dyn, DynArgs, DynSpec, LayoutError, Result, Spec, TakeCursor, Vec3, Vec3View, Vec3ViewMut,
    };
}
