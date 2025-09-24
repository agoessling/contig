//! `#[contig]` derive: generates config/layout/view types plus a `Contig` impl
//! for concrete user structs.
//!
//! The macro requires `#[contig(scalar = <ty>)]` to specify the scalar type (e.g. `f64`).
//! It only supports non-generic structs with named fields; per-field `#[contig(...)]`
//! attributes determine whether a field is dynamic and what runtime arguments it needs.

use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Attribute, Data, DeriveInput, Fields, MetaNameValue, Token, Type, parse::Parser,
    parse_macro_input, parse_quote, punctuated::Punctuated, spanned::Spanned,
};

/// Clone a field, removing any `#[contig(...)]` helper attributes so they are
/// not re-emitted in the generated struct definition.
fn strip_contig_attrs(field: &syn::Field) -> syn::Field {
    let mut clone = field.clone();
    clone.attrs.retain(|attr| !attr.path().is_ident("contig"));
    clone
}

/// Parse the scalar type from the attribute arguments (`#[contig(scalar = ...)]`).
fn parse_scalar_type(attr: TokenStream) -> syn::Result<Type> {
    let parser = Punctuated::<MetaNameValue, Token![,]>::parse_terminated;
    let args = parser
        .parse2(attr.into())
        .map_err(|e| syn::Error::new(e.span(), "invalid #[contig] arguments"))?;

    for nv in args {
        if nv.path.is_ident("scalar") {
            let ty_tokens = nv.value.to_token_stream();
            return syn::parse2::<Type>(ty_tokens).map_err(|err| {
                syn::Error::new(err.span(), "scalar must be a type path (e.g., f64)")
            });
        }
    }

    Err(syn::Error::new_spanned(
        quote! { #[contig(scalar = <ty>)] },
        "missing `scalar` attribute: use #[contig(scalar = f64)]",
    ))
}

/// Field-level helper attributes are currently parsed but unused; this stub
/// remains so we can extend handling later without touching call sites.
fn parse_flags(attrs: &[Attribute]) {
    let _ = attrs;
}

/// Expand a struct annotated with `#[contig(...)]` into a fully operational
/// configuration/layout/view trio plus a [`contig_core::Contig`] implementation.
///
/// ```
/// use contig_derive::contig;
///
/// #[contig(scalar = f64)]
/// struct PointMass {
///     mass: f64,
///     bias: f64,
/// }
///
/// let cfg = PointMassCfg { mass: (), bias: () };
/// let layout = PointMassLayout::from_config(&cfg).unwrap();
/// let mut buffer = vec![0.0; layout.len()];
/// {
///     let mut view = layout.view(buffer.as_mut_slice());
///     *view.mass() = 12.0;
///     *view.bias() = 0.5;
/// }
/// let view = layout.cview(buffer.as_slice());
/// assert_eq!(*view.mass(), 12.0);
/// assert_eq!(*view.bias(), 0.5);
/// ```
///
/// The macro preserves the user-written struct (minus helper attributes) and
/// emits sibling `Cfg`, `Layout`, `View`, and `ConstView` types alongside a
/// [`contig_core::Contig`] implementation.
#[proc_macro_attribute]
pub fn contig(attr: TokenStream, item: TokenStream) -> TokenStream {
    let scalar_ty = match parse_scalar_type(attr) {
        Ok(ty) => ty,
        Err(err) => return err.to_compile_error().into(),
    };

    let input = parse_macro_input!(item as DeriveInput);

    if !input.generics.params.is_empty() {
        return syn::Error::new(
            input.generics.span(),
            "#[contig] does not support generic structs; instantiate a concrete type",
        )
        .to_compile_error()
        .into();
    }

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

    let vis = input.vis.clone();
    let retained_attrs: Vec<Attribute> = input
        .attrs
        .iter()
        .filter(|attr| !attr.path().is_ident("contig"))
        .cloned()
        .collect();

    let cfg_ident = format_ident!("{}Cfg", struct_ident);
    let layout_ident = format_ident!("{}Layout", struct_ident);
    let view_ident = format_ident!("{}View", struct_ident);
    let cview_ident = format_ident!("{}ConstView", struct_ident);
    let struct_name = struct_ident.to_string();

    let cleaned_fields: Vec<syn::Field> = fields.iter().map(strip_contig_attrs).collect();

    let mut cfg_fields = Vec::new();
    let mut layout_struct_fields = Vec::new();
    let mut layout_inits = Vec::new();
    let mut layout_builders = Vec::new();
    let mut view_methods_mut = Vec::new();
    let mut view_methods_const = Vec::new();
    let mut contig_bounds = Vec::<syn::WherePredicate>::new();

    for field in fields.iter() {
        parse_flags(&field.attrs);
        let fname = field.ident.clone().expect("named field");
        let fty = &field.ty;
        let off_ident = format_ident!("off_{}", fname);
        let lay_ident = format_ident!("layout_{}", fname);
        let fname_str = fname.to_string();
        let cfg_field_doc = format!(
            "Runtime configuration for `{}::{}`.",
            struct_name.as_str(),
            fname_str
        );
        let offset_doc = format!(
            "Scalar range covering `{}::{}` inside the buffer.",
            struct_name.as_str(),
            fname_str
        );
        let layout_field_doc = format!(
            "Layout metadata for `{}::{}`.",
            struct_name.as_str(),
            fname_str
        );
        let mut_method_doc = format!(
            "Borrow a mutable view into `{}::{}`.",
            struct_name.as_str(),
            fname_str
        );
        let const_method_doc = format!(
            "Borrow a read-only view into `{}::{}`.",
            struct_name.as_str(),
            fname_str
        );

        cfg_fields.push(quote! {
            #[doc = #cfg_field_doc]
            pub #fname: <#fty as contig_core::Contig<#scalar_ty>>::Config
        });

        layout_struct_fields.push(quote! {
            #[doc = #offset_doc]
            pub #off_ident: core::ops::Range<usize>
        });
        layout_struct_fields.push(quote! {
            #[doc = #layout_field_doc]
            pub #lay_ident: <#fty as contig_core::Contig<#scalar_ty>>::Layout
        });

        layout_inits.push(quote! { #off_ident });
        layout_inits.push(quote! { #lay_ident });

        layout_builders.push(quote! {
            let #lay_ident = <#fty as contig_core::Contig<#scalar_ty>>::layout(&cfg.#fname)?;
            let #off_ident = __cursor
                .take_range(<#fty as contig_core::Contig<#scalar_ty>>::len(&#lay_ident));
        });

        view_methods_mut.push(quote! {
            #[doc = #mut_method_doc]
            pub fn #fname(&mut self) -> <#fty as contig_core::Contig<#scalar_ty>>::MutView<'_> {
                <#fty as contig_core::Contig<#scalar_ty>>::view_mut(
                    &self.layout.#lay_ident,
                    &mut self.base[self.layout.#off_ident.clone()],
                )
            }
        });
        view_methods_const.push(quote! {
            #[doc = #const_method_doc]
            pub fn #fname(&self) -> <#fty as contig_core::Contig<#scalar_ty>>::ConstView<'_> {
                <#fty as contig_core::Contig<#scalar_ty>>::view(
                    &self.layout.#lay_ident,
                    &self.base[self.layout.#off_ident.clone()],
                )
            }
        });

        contig_bounds.push(parse_quote! {
            #fty: contig_core::Contig<#scalar_ty>
        });
    }

    let cfg_doc = format!(
        "Runtime configuration for `{}` produced by `#[contig]`.",
        struct_name.as_str()
    );
    let layout_doc = format!(
        "Layout metadata for `{}` computed by `#[contig]`.",
        struct_name.as_str()
    );
    let layout_len_doc = "Total scalar elements spanned by this layout.";
    let view_doc = format!(
        "Mutable view over `{}` borrowed from a contiguous buffer.",
        struct_name.as_str()
    );
    let cview_doc = format!(
        "Read-only view over `{}` borrowed from a contiguous buffer.",
        struct_name.as_str()
    );
    let view_as_mut_slice_doc = "Expose the underlying mutable slice backing this view.";
    let const_view_as_slice_doc = "Expose the underlying immutable slice backing this view.";
    let layout_from_config_doc = format!(
        "Compute the layout for `{}` from its configuration.",
        struct_name.as_str()
    );
    let layout_len_method_doc = "Total scalar footprint of this layout.";
    let layout_view_doc = "Create a mutable view into the supplied buffer.";
    let layout_cview_doc = "Create a read-only view into the supplied buffer.";

    let struct_definition = {
        let attrs = &retained_attrs;
        quote! {
            #( #attrs )*
            #vis struct #struct_ident {
                #( #cleaned_fields ),*
            }
        }
    };

    let cfg_definition = quote! {
        #[doc = #cfg_doc]
        #[derive(Clone)]
        #vis struct #cfg_ident {
            #( #cfg_fields, )*
        }
    };

    let layout_definition = quote! {
        #[doc = #layout_doc]
        #[derive(Clone)]
        #vis struct #layout_ident {
            #( #layout_struct_fields, )*
            #[doc = #layout_len_doc]
            pub len: usize,
        }
    };

    let layout_impl = quote! {
        impl #layout_ident {
            #[doc = #layout_from_config_doc]
            pub fn from_config(cfg: &#cfg_ident) -> contig_core::Result<Self> {
                let mut __cursor = contig_core::TakeCursor::new();
                #( #layout_builders )*
                let len = __cursor.finish();
                Ok(Self {
                    #( #layout_inits, )*
                    len,
                })
            }

            #[inline]
            #[doc = #layout_len_method_doc]
            pub fn len(&self) -> usize {
                self.len
            }

            #[doc = #layout_view_doc]
            pub fn view<'a>(
                &'a self,
                base: &'a mut [#scalar_ty],
            ) -> #view_ident<'a>
            where
                #scalar_ty: 'a,
            {
                assert!(base.len() >= self.len, "buffer too small for layout");
                #view_ident { base, layout: self }
            }

            #[doc = #layout_cview_doc]
            pub fn cview<'a>(
                &'a self,
                base: &'a [#scalar_ty],
            ) -> #cview_ident<'a>
            where
                #scalar_ty: 'a,
            {
                assert!(base.len() >= self.len, "buffer too small for layout");
                #cview_ident { base, layout: self }
            }
        }
    };

    let view_definition = quote! {
        #[doc = #view_doc]
        #vis struct #view_ident<'a> {
            base: &'a mut [#scalar_ty],
            layout: &'a #layout_ident,
        }
    };

    let const_view_definition = quote! {
        #[doc = #cview_doc]
        #vis struct #cview_ident<'a> {
            base: &'a [#scalar_ty],
            layout: &'a #layout_ident,
        }
    };

    let view_impl = quote! {
        impl<'a> #view_ident<'a> {
            #[inline]
            #[doc = #view_as_mut_slice_doc]
            pub fn as_mut_slice(&mut self) -> &mut [#scalar_ty] {
                self.base
            }
            #( #view_methods_mut )*
        }
    };

    let const_view_impl = quote! {
        impl<'a> #cview_ident<'a> {
            #[inline]
            #[doc = #const_view_as_slice_doc]
            pub fn as_slice(&self) -> &[#scalar_ty] {
                self.base
            }
            #( #view_methods_const )*
        }
    };

    let const_view_type = quote! { #cview_ident<'a> };
    let view_type = quote! { #view_ident<'a> };

    let contig_where_clause = if contig_bounds.is_empty() {
        None
    } else {
        let preds = contig_bounds.iter();
        Some(quote! { where #( #preds ),* })
    };

    let contig_impl = quote! {
        impl contig_core::Contig<#scalar_ty> for #struct_ident #contig_where_clause {
            type Config = #cfg_ident;
            type Layout = #layout_ident;
            type ConstView<'a> = #const_view_type;
            type MutView<'a> = #view_type;

            fn layout(config: &Self::Config) -> contig_core::Result<Self::Layout> {
                #layout_ident::from_config(config)
            }

            fn len(layout: &Self::Layout) -> usize {
                layout.len()
            }

            fn view<'a>(
                layout: &'a Self::Layout,
                buf: &'a [#scalar_ty],
            ) -> Self::ConstView<'a> {
                layout.cview(buf)
            }

            fn view_mut<'a>(
                layout: &'a Self::Layout,
                buf: &'a mut [#scalar_ty],
            ) -> Self::MutView<'a> {
                layout.view(buf)
            }
        }
    };

    let expanded = quote! {
        #struct_definition
        #cfg_definition
        #layout_definition
        #layout_impl
        #view_definition
        #const_view_definition
        #view_impl
        #const_view_impl
        #contig_impl
    };

    expanded.into()
}
