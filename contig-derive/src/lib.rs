use proc_macro::TokenStream;
use quote::{ToTokens, format_ident, quote};
use syn::{
    Attribute, Data, DeriveInput, Fields, MetaNameValue, Token, Type, parse::Parser,
    parse_macro_input, parse_quote, punctuated::Punctuated, spanned::Spanned,
};

fn strip_contig_attrs(field: &syn::Field) -> syn::Field {
    let mut clone = field.clone();
    clone.attrs.retain(|attr| !attr.path().is_ident("contig"));
    clone
}

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

fn parse_flags(attrs: &[Attribute]) {
    let _ = attrs;
}

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

        cfg_fields.push(quote! {
            pub #fname: <#fty as contig_core::Contig<#scalar_ty>>::Config
        });

        layout_struct_fields.push(quote! {
            pub #off_ident: core::ops::Range<usize>
        });
        layout_struct_fields.push(quote! {
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
            pub fn #fname(&mut self) -> <#fty as contig_core::Contig<#scalar_ty>>::MutView<'_> {
                <#fty as contig_core::Contig<#scalar_ty>>::view_mut(
                    &self.layout.#lay_ident,
                    &mut self.base[self.layout.#off_ident.clone()],
                )
            }
        });
        view_methods_const.push(quote! {
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
        #[derive(Clone)]
        #vis struct #cfg_ident {
            #( #cfg_fields, )*
        }
    };

    let layout_definition = quote! {
        #[derive(Clone)]
        #vis struct #layout_ident {
            #( #layout_struct_fields, )*
            pub len: usize,
        }
    };

    let layout_impl = quote! {
        impl #layout_ident {
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
            pub fn len(&self) -> usize {
                self.len
            }

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
        #vis struct #view_ident<'a> {
            base: &'a mut [#scalar_ty],
            layout: &'a #layout_ident,
        }
    };

    let const_view_definition = quote! {
        #vis struct #cview_ident<'a> {
            base: &'a [#scalar_ty],
            layout: &'a #layout_ident,
        }
    };

    let view_impl = quote! {
        impl<'a> #view_ident<'a> {
            #[inline]
            pub fn as_mut_slice(&mut self) -> &mut [#scalar_ty] {
                self.base
            }
            #( #view_methods_mut )*
        }
    };

    let const_view_impl = quote! {
        impl<'a> #cview_ident<'a> {
            #[inline]
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
