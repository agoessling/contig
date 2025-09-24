use contig_core::{Contig, Result};
use core::marker::PhantomData;

/// Marker type representing a fixed `[F; 3]` contiguous vector.
#[derive(Clone, Copy, Debug)]
pub struct Vec3<F>(PhantomData<F>);

/// Read-only view across three consecutive scalars laid out as `x`, `y`, `z`.
#[derive(Clone, Copy, Debug)]
pub struct Vec3View<'a, F> {
    slice: &'a [F],
}

/// Mutable view across three consecutive scalars laid out as `x`, `y`, `z`.
#[derive(Debug)]
pub struct Vec3ViewMut<'a, F> {
    slice: &'a mut [F],
}

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
