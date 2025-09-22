//! #[contig] derive macro (nested config only, explicit flags)
//!
//! Field-level flags:
//!   - #[contig(len)]           // this node has a runtime length
//!   - #[contig(shape)]         // this node has a runtime matrix shape (rows, cols)
//!   - #[contig(elem_len)]      // the ELEMENT has a runtime length
//!   - #[contig(elem_shape)]    // the ELEMENT has a runtime shape
//!
//! Examples:
//!   mats: sc::Dyn<[sc::NaDMatrix<F>]> #[contig(len, elem_shape)]
//!   coeffs: [sc::NaDVector<F>; 3]     #[contig(elem_len)]
//!
//! This macro generates:
//!   <Type>Cfg          (top-level config, nested)
//!   <Type>_<field>_Cfg (+ _elem_Cfg as needed) per dynamic field
//!   <Type>Layout, <Type>View<'a, F>, <Type>ConstView<'a, F>
//!
//! The macro delegates per-type view construction to contig-core::Spec/DynSpec.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input,
    spanned::Spanned,
    Attribute, Data, DeriveInput, Fields,
};

#[derive(Default, Clone, Copy)]
struct Flags {
    len: bool,
    shape: bool,
    elem_len: bool,
    elem_shape: bool,
}

fn parse_flags(attrs: &[Attribute]) -> syn::Result<Flags> {
    let mut f = Flags::default();
    for attr in attrs {
        if !attr.path().is_ident("contig") {
            continue;
        }
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("len") {
                f.len = true;
            } else if meta.path.is_ident("shape") {
                f.shape = true;
            } else if meta.path.is_ident("elem_len") {
                f.elem_len = true;
            } else if meta.path.is_ident("elem_shape") {
                f.elem_shape = true;
            }
            Ok(())
        })?;
    }
    Ok(f)
}

#[proc_macro_attribute]
pub fn contig(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);

    let struct_ident = input.ident.clone();
    let data = match &input.data {
        Data::Struct(ds) => ds,
        _ => {
            return syn::Error::new(input.span(), "#[contig] supports only structs")
                .to_compile_error()
                .into();
        }
    };
    let fields = match &data.fields {
        Fields::Named(named) => &named.named,
        _ => {
            return syn::Error::new(input.span(), "#[contig] requires named fields")
                .to_compile_error()
                .into();
        }
    };

    // Generated identifiers
    let cfg_ident = format_ident!("{}Cfg", struct_ident);
    let layout_ident = format_ident!("{}Layout", struct_ident);
    let view_ident = format_ident!("{}View", struct_ident);
    let cview_ident = format_ident!("{}ConstView", struct_ident);

    // Buckets to accumulate code pieces
    let mut node_cfg_structs = Vec::<proc_macro2::TokenStream>::new(); // per-field cfg structs
    let mut top_cfg_fields = Vec::<proc_macro2::TokenStream>::new(); // fields in <Type>Cfg
    let mut layout_fields = Vec::<proc_macro2::TokenStream>::new(); // fields in <Type>Layout
    let mut layout_builders = Vec::<proc_macro2::TokenStream>::new(); // code in Layout::from_config
    let mut view_methods_mut = Vec::<proc_macro2::TokenStream>::new(); // methods on <Type>View
    let mut cview_methods = Vec::<proc_macro2::TokenStream>::new(); // methods on <Type>ConstView

    for field in fields.iter() {
        let fname = field.ident.clone().unwrap();
        let fty = &field.ty;
        let flags = match parse_flags(&field.attrs) {
            Ok(f) => f,
            Err(e) => return e.to_compile_error().into(),
        };

        let off_ident = format_ident!("off_{}", fname);
        let node_cfg_ident = format_ident!("{}_{}_Cfg", struct_ident, fname);
        let elem_cfg_ident = format_ident!("{}_{}_elem_Cfg", struct_ident, fname);

        // Determine whether this field is dynamic (needs a cfg node)
        let dynamic_node = flags.len || flags.shape || flags.elem_len || flags.elem_shape;

        if dynamic_node {
            // --- Node cfg definition ---
            let mut node_cfg_fields = Vec::<proc_macro2::TokenStream>::new();
            if flags.len {
                node_cfg_fields.push(quote! { pub len: usize });
            }
            if flags.shape {
                node_cfg_fields.push(quote! { pub shape: (usize, usize) });
            }

            // Optional element cfg
            let mut maybe_elem_struct = None::<proc_macro2::TokenStream>;
            let mut maybe_elem_field = None::<proc_macro2::TokenStream>;
            if flags.elem_len || flags.elem_shape {
                let mut ef = Vec::<proc_macro2::TokenStream>::new();
                if flags.elem_len {
                    ef.push(quote! { pub len: usize });
                }
                if flags.elem_shape {
                    ef.push(quote! { pub shape: (usize, usize) });
                }
                maybe_elem_struct = Some(quote! { pub struct #elem_cfg_ident { #( #ef, )* } });
                maybe_elem_field = Some(quote! { pub elem: #elem_cfg_ident });
            }
            if let Some(f) = maybe_elem_field {
                node_cfg_fields.push(f);
            }

            node_cfg_structs
                .push(quote! { pub struct #node_cfg_ident { #( #node_cfg_fields, )* } });
            if let Some(es) = maybe_elem_struct {
                node_cfg_structs.push(es);
            }

            // --- Top-level cfg field includes this node cfg ---
            top_cfg_fields.push(quote! { pub #fname: #node_cfg_ident });

            // --- Layout: keep the range and the node cfg (cached) ---
            layout_fields.push(quote! {
                pub #off_ident: core::ops::Range<usize>,
                pub #fname: #node_cfg_ident,
            });

            // --- Layout builder: compute footprint using DynSpec::<F> ---
            // Build args and elem_args from cfg
            let cfg_len_expr = if flags.len {
                quote! { Some(cfg.#fname.len) }
            } else {
                quote! { None }
            };
            let cfg_shape_expr = if flags.shape {
                quote! { Some(cfg.#fname.shape) }
            } else {
                quote! { None }
            };
            let cfg_args_expr = quote! {
                contig_core::DynArgs {
                    len: #cfg_len_expr,
                    shape: #cfg_shape_expr,
                }
            };
            let has_elem_args = flags.elem_len || flags.elem_shape;
            let cfg_elem_args_expr = if has_elem_args {
                let elem_len_expr = if flags.elem_len {
                    quote! { Some(cfg.#fname.elem.len) }
                } else {
                    quote! { None }
                };
                let elem_shape_expr = if flags.elem_shape {
                    quote! { Some(cfg.#fname.elem.shape) }
                } else {
                    quote! { None }
                };
                quote! {
                    contig_core::DynArgs {
                        len: #elem_len_expr,
                        shape: #elem_shape_expr,
                    }
                }
            } else {
                quote! { contig_core::DynArgs::default() }
            };
            let dyn_len_expr = if has_elem_args {
                quote! { <#fty as contig_core::DynSpec<F>>::dyn_len_with_elem(&__args, &__elem_args) }
            } else {
                quote! { <#fty as contig_core::DynSpec<F>>::dyn_len(&__args) }
            };

            let build = if flags.len || flags.shape {
                quote! {
                    let __args = #cfg_args_expr;
                    let __elem_args = #cfg_elem_args_expr;
                    let __n = #dyn_len_expr;
                    let #off_ident = __cursor.take_range(__n);
                }
            } else {
                let msg = "Element-only dynamic flags without node len/shape require parent static sizing.";
                quote! { compile_error!(#msg); }
            };
            layout_builders.push(build);

            // --- Accessors: delegate to DynSpec::mview_full/cview_full ---
            let layout_len_expr = if flags.len {
                quote! { Some(self.layout.#fname.len) }
            } else {
                quote! { None }
            };
            let layout_shape_expr = if flags.shape {
                quote! { Some(self.layout.#fname.shape) }
            } else {
                quote! { None }
            };
            let layout_args_expr = quote! {
                contig_core::DynArgs {
                    len: #layout_len_expr,
                    shape: #layout_shape_expr,
                }
            };
            let layout_elem_args_expr = if has_elem_args {
                let elem_len_expr = if flags.elem_len {
                    quote! { Some(self.layout.#fname.elem.len) }
                } else {
                    quote! { None }
                };
                let elem_shape_expr = if flags.elem_shape {
                    quote! { Some(self.layout.#fname.elem.shape) }
                } else {
                    quote! { None }
                };
                quote! {
                    contig_core::DynArgs {
                        len: #elem_len_expr,
                        shape: #elem_shape_expr,
                    }
                }
            } else {
                quote! { contig_core::DynArgs::default() }
            };
            let make_args = quote! {
                let __args = #layout_args_expr;
                let __elem_args = #layout_elem_args_expr;
            };
            view_methods_mut.push(quote! {
                pub fn #fname(&mut self) -> <#fty as contig_core::DynSpec<F>>::MView<'_> {
                    #make_args
                    <#fty as contig_core::DynSpec<F>>::mview_full(
                        &mut self.base[self.layout.#off_ident.clone()], &__args, &__elem_args)
                }
            });
            cview_methods.push(quote! {
                pub fn #fname(&self) -> <#fty as contig_core::DynSpec<F>>::CView<'_> {
                    #make_args
                    <#fty as contig_core::DynSpec<F>>::cview_full(
                        &self.base[self.layout.#off_ident.clone()], &__args, &__elem_args)
                }
            });
        } else {
            // --- Static node: Spec<F> ---
            layout_fields.push(quote! {
                pub #off_ident: core::ops::Range<usize>,
            });
            layout_builders.push(quote! {
                let #off_ident = __cursor.take_range(<#fty as contig_core::Spec<F>>::STATIC_LEN);
            });
            view_methods_mut.push(quote! {
                pub fn #fname(&mut self) -> <#fty as contig_core::Spec<F>>::MView<'_> {
                    <#fty as contig_core::Spec<F>>::mview(&mut self.base[self.layout.#off_ident.clone()])
                }
            });
            cview_methods.push(quote! {
                pub fn #fname(&self) -> <#fty as contig_core::Spec<F>>::CView<'_> {
                    <#fty as contig_core::Spec<F>>::cview(&self.base[self.layout.#off_ident.clone()])
                }
            });
        }
    }

    // Build top-level cfg struct
    let top_cfg = quote! { pub struct #cfg_ident { #( #top_cfg_fields, )* } };

    // Final assembly
    let expanded = quote! {
        // re-emit user struct
        #input

        // node cfgs
        #( #node_cfg_structs )*

        // top-level cfg
        #top_cfg

        // layout
        pub struct #layout_ident {
            #( #layout_fields )*
            pub len: usize,
        }

        impl #layout_ident {
            pub fn from_config<F>(cfg: &#cfg_ident) -> contig_core::Result<Self> {
                let mut __cursor = contig_core::TakeCursor::new();
                #( #layout_builders )*
                let len = __cursor.finish();
                Ok(Self {
                    #( #layout_fields )*
                    len,
                })
            }

            #[inline] pub fn len(&self) -> usize { self.len }

            pub fn view<'a, F>(&'a self, base: &'a mut [F]) -> #view_ident<'a, F> {
                assert!(base.len() >= self.len, "buffer too small for layout");
                #view_ident { base, layout: self }
            }
            pub fn cview<'a, F>(&'a self, base: &'a [F]) -> #cview_ident<'a, F> {
                assert!(base.len() >= self.len, "buffer too small for layout");
                #cview_ident { base, layout: self }
            }
        }

        // views
        pub struct #view_ident<'a, F> { base: &'a mut [F], layout: &'a #layout_ident }
        pub struct #cview_ident<'a, F> { base: &'a [F], layout: &'a #layout_ident }

        impl<'a, F> #view_ident<'a, F> {
            #[inline] pub fn as_mut_slice(&mut self) -> &mut [F] { self.base }
            #( #view_methods_mut )*
        }
        impl<'a, F> #cview_ident<'a, F> {
            #[inline] pub fn as_slice(&self) -> &[F] { self.base }
            #( #cview_methods )*
        }
    };

    expanded.into()
}
