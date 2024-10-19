//! This contains more types for testing the `rust_type_name` module,
//! using parts of the rustdoc output (e.g. function parameters) that
//! are not currently exposed by `trustfall-rustdoc-adapter`.

pub trait GAT<T> {
    type Type<'a, U>
    where
        Self: 'a,
        U: 'a;
}

pub struct Struct<'a>(&'a ());

impl<'a> GAT<&'a ()> for Struct<'a> {
    type Type<'b, U> = Result<&'a (), &'b U> where Self: 'b, U: 'b;
}

pub struct Constant<const N: usize>;

const fn const_fn() -> usize {
    let mut i = 0;

    while i < 27 {
        i += 5;
    }

    i
}

/// Tests for base-case variants of [`Type`]
///
/// [`Type`]: https://docs.rs/rustdoc-types/latest/rustdoc_types/enum.Type.html
pub struct TypeEnum<T> {
    pub resolved_path: Option<()>,
    pub dyn_trait: Box<dyn std::io::Read>,
    pub generic: T,
    pub primitive: u32,
    pub function_pointer: fn(()) -> i32,
    pub tuple: (u32, (), T),
    pub slice: Box<[u8]>,
    pub array: [(); 1 + 2],
    pub array2: [(); usize::MAX >> (usize::BITS - 1)],
    pub array3: [(); const_fn()],
    // impl Trait
    // Infer
    pub raw_pointer: *const u8,
    pub borrowed_ref: &'static str,
    pub qualified_path: <std::str::SplitAsciiWhitespace<'static> as Iterator>::Item,
}

pub fn is_synthetic(x: impl std::any::Any) -> impl std::any::Any {
    x
}

pub fn dyn_ambiguity<'a>(
    a: &'a (impl Fn() -> *const fn() -> &'a (dyn Iterator<Item = ()> + Unpin) + Send),
    b: Box<dyn Fn() -> *const (dyn Unpin + Fn() -> &'static mut (dyn std::any::Any + Sync)) + Sync>,
    c: fn() -> &'a (dyn Send + Fn() -> *const dyn std::any::Any),
    no_parens: impl for<'x> Fn(&'x ()) -> &'x dyn std::fmt::Debug,
    sanity: &dyn std::fmt::Display,
) {
}

pub fn my_generic_function<'a, T, U: GAT<T>>(
    a: &'a &'static mut *const T,
    b: &(dyn Iterator<Item = T> + Unpin + Send),
    c: Constant<25>,
    d: impl for<'x> FnMut(
            &'a unsafe extern "C" fn(
                *const [u8],
                &'x mut *mut (),
                ...
            ) -> std::borrow::Cow<'static, [u8]>,
        ) -> &'x (dyn std::fmt::Display)
        + Send
        + 'static,
    e: <U as GAT<T>>::Type<'a, &'static *const ()>,
) -> impl std::future::Future<Output: Iterator<Item: 'a + Send> + for<'z> FnMut(&'z ()) -> &'z &'a ()>
{
    unimplemented!()
}
