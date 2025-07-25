pub(crate) mod drop;
pub(super) mod list;
pub(crate) mod polars_extension;

use std::mem;

use arrow::array::FixedSizeBinaryArray;
use arrow::bitmap::BitmapBuilder;
use arrow::buffer::Buffer;
use arrow::datatypes::ExtensionType;
use polars_extension::PolarsExtension;
use polars_utils::format_pl_smallstr;
use polars_utils::relaxed_cell::RelaxedCell;

use crate::PROCESS_ID;
use crate::prelude::*;

static POLARS_ALLOW_EXTENSION: RelaxedCell<bool> = RelaxedCell::new_bool(false);

/// Control whether extension types may be created.
///
/// If the environment variable POLARS_ALLOW_EXTENSION is set, this function has no effect.
pub fn set_polars_allow_extension(toggle: bool) {
    POLARS_ALLOW_EXTENSION.store(toggle)
}

/// Invariants
/// `ptr` must point to start a `T` allocation
/// `n_t_vals` must represent the correct number of `T` values in that allocation
unsafe fn create_drop<T: Sized>(mut ptr: *const u8, n_t_vals: usize) -> Box<dyn FnMut()> {
    Box::new(move || {
        let t_size = size_of::<T>() as isize;
        for _ in 0..n_t_vals {
            let _ = std::ptr::read_unaligned(ptr as *const T);
            ptr = ptr.offset(t_size)
        }
    })
}

#[allow(clippy::type_complexity)]
struct ExtensionSentinel {
    drop_fn: Option<Box<dyn FnMut()>>,
    // A function on the heap that take a `array: FixedSizeBinary` and a `name: PlSmallStr`
    // and returns a `Series` of `ObjectChunked<T>`
    pub(crate) to_series_fn: Option<Box<dyn Fn(&FixedSizeBinaryArray, &PlSmallStr) -> Series>>,
}

impl Drop for ExtensionSentinel {
    fn drop(&mut self) {
        let mut drop_fn = self.drop_fn.take().unwrap();
        drop_fn()
    }
}

// https://stackoverflow.com/questions/28127165/how-to-convert-struct-to-u8d
// not entirely sure if padding bytes in T are initialized or not.
unsafe fn any_as_u8_slice<T: Sized>(p: &T) -> &[u8] {
    std::slice::from_raw_parts((p as *const T) as *const u8, size_of::<T>())
}

/// Create an extension Array that can be sent to arrow and (once wrapped in [`PolarsExtension`] will
/// also call drop on `T`, when the array is dropped.
pub(crate) fn create_extension<I: Iterator<Item = Option<T>> + TrustedLen, T: Sized + Default>(
    iter: I,
) -> PolarsExtension {
    let env = "POLARS_ALLOW_EXTENSION";
    if !(POLARS_ALLOW_EXTENSION.load() || std::env::var(env).is_ok()) {
        panic!("creating extension types not allowed - try setting the environment variable {env}")
    }
    let t_size = size_of::<T>();
    let t_alignment = align_of::<T>();
    let n_t_vals = iter.size_hint().1.unwrap();

    let mut buf = Vec::with_capacity(n_t_vals * t_size);
    let mut validity = BitmapBuilder::with_capacity(n_t_vals);

    // when we transmute from &[u8] to T, T must be aligned correctly,
    // so we pad with bytes until the alignment matches
    let n_padding = (buf.as_ptr() as usize) % t_alignment;
    buf.extend(std::iter::repeat_n(0, n_padding));

    // transmute T as bytes and copy in buffer
    for opt_t in iter.into_iter() {
        match opt_t {
            Some(t) => {
                unsafe {
                    buf.extend_from_slice(any_as_u8_slice(&t));
                    // SAFETY: we allocated upfront
                    validity.push_unchecked(true)
                }
                mem::forget(t);
            },
            None => {
                unsafe {
                    buf.extend_from_slice(any_as_u8_slice(&T::default()));
                    // SAFETY: we allocated upfront
                    validity.push_unchecked(false)
                }
            },
        }
    }

    // we slice the buffer because we want to ignore the padding bytes from here
    // they can be forgotten
    let buf: Buffer<u8> = buf.into();
    let len = buf.len() - n_padding;
    let buf = buf.sliced(n_padding, len);

    // ptr to start of T, not to start of padding
    let ptr = buf.as_slice().as_ptr();

    // SAFETY:
    // ptr and t are correct
    let drop_fn = unsafe { create_drop::<T>(ptr, n_t_vals) };
    let et = Box::new(ExtensionSentinel {
        drop_fn: Some(drop_fn),
        to_series_fn: None,
    });
    let et_ptr = &*et as *const ExtensionSentinel;
    std::mem::forget(et);

    let metadata = format_pl_smallstr!("{};{}", *PROCESS_ID, et_ptr as usize);

    let physical_type = ArrowDataType::FixedSizeBinary(t_size);
    let extension_type = ArrowDataType::Extension(Box::new(ExtensionType {
        name: PlSmallStr::from_static(EXTENSION_NAME),
        inner: physical_type,
        metadata: Some(metadata),
    }));

    let array = FixedSizeBinaryArray::new(extension_type, buf, validity.into_opt_validity());

    // SAFETY: we just heap allocated the ExtensionSentinel, so its alive.
    unsafe { PolarsExtension::new(array) }
}

#[cfg(test)]
mod test {
    use std::fmt::{Display, Formatter};
    use std::hash::{Hash, Hasher};

    use polars_utils::total_ord::TotalHash;

    use super::*;

    #[derive(Clone, Debug, Default, Eq, Hash, PartialEq)]
    struct Foo {
        pub a: i32,
        pub b: u8,
        pub other_heap: String,
    }

    impl TotalEq for Foo {
        fn tot_eq(&self, other: &Self) -> bool {
            self == other
        }
    }

    impl TotalHash for Foo {
        fn tot_hash<H>(&self, state: &mut H)
        where
            H: Hasher,
        {
            self.hash(state);
        }
    }

    impl Display for Foo {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            write!(f, "{self:?}")
        }
    }

    impl PolarsObject for Foo {
        fn type_name() -> &'static str {
            "object"
        }
    }

    #[test]
    fn test_create_extension() {
        set_polars_allow_extension(true);
        // Run this under MIRI.
        let foo = Foo {
            a: 1,
            b: 1,
            other_heap: "foo".into(),
        };
        let foo2 = Foo {
            a: 1,
            b: 1,
            other_heap: "bar".into(),
        };

        let vals = vec![Some(foo), Some(foo2)];
        create_extension(vals.into_iter());
    }
}
