#[test]
fn ui() {
    let t = trybuild::TestCases::new();
    t.pass("tests/ui/basic.rs");
    t.pass("tests/ui/nested.rs");
    t.compile_fail("tests/ui/missing_scalar.rs");
    t.compile_fail("tests/ui/generic_struct.rs");
    t.compile_fail("tests/ui/enum_not_allowed.rs");
    t.compile_fail("tests/ui/wrong_field_attr.rs");
    t.compile_fail("tests/ui/missing_contig_import.rs");
}
