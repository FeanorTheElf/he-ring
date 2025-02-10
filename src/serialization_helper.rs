use std::{alloc::Allocator, marker::PhantomData};

use feanor_math::seq::VectorView;
use serde::de::{DeserializeSeed, IgnoredAny, SeqAccess, Visitor};
use serde::{Deserializer, Serialize};

///
/// Macro to implement [`serde::de::DeserializeSeed`] for a custom type.
/// 
/// More concretely, when using this macro, you will define a struct and
/// a [`DeserializeSeed`] for each of its fields (which can access data from
/// your custom type). The custom type can then be used to deserialize the
/// struct, by deserializing each of its fields separately with the derived
/// [`DeserializeSeed`]s.
/// 
/// # Example
/// 
/// As a very simple example, this macro can be used as a poor man's version
/// of `#[derive(Deserialize)]` as follows.
/// 
/// The function `deserializer()` in
/// ```
/// # use he_ring::impl_deserialize_seed_for_dependent_struct;
/// # use serde::*;
/// # use std::marker::PhantomData;
/// struct FooDeserializeSeed;
/// impl_deserialize_seed_for_dependent_struct!{
///     struct Foo<'de> using FooDeserializeSeed {
///         a: i64: |_| PhantomData::<i64>,
///         b: String: |_| PhantomData::<String>
///     }
/// }
/// fn deserializer<'de>() -> impl serde::de::DeserializeSeed<'de, Value = Foo<'de>> {
///     FooDeserializeSeed
/// }
/// ```
/// is roughly equivalent to `deserializer()` as in
/// ```
/// # use serde::*;
/// # use std::marker::PhantomData;
/// #[derive(Deserialize)]
/// struct Foo {
///     a: i64,
///     b: String
/// }
/// fn deserializer<'de>() -> impl serde::de::DeserializeSeed<'de, Value = Foo> {
///     PhantomData::<Foo>
/// }
/// ```
/// 
/// It becomes more interesting if fields of the result struct should be deserialized
/// using a [`DeserializeSeed`], since in this case, it cannot be achieved using `#[derive(Deserialize)]`
/// anymore. Note however that [`impl_deserialize_seed_for_dependent_struct!`] can only implement
/// [`DeserializeSeed`] for a type in terms of more basic [`DeserializeSeed`]s. Hence, the leaves of the
/// "deserialization-tree" must still be implemented manually (this is also the case for `#[derive(Deserialize)]`
/// of course, but the leaves here are usually std type `i64`, `&[u8]` or `String`, for which the implementation
/// of [`serde::Deserialize`] is contained in `serde`).
/// ```
/// # use he_ring::impl_deserialize_seed_for_dependent_struct;
/// # use serde::*;
/// # use serde::de::DeserializeSeed;
/// # use std::marker::PhantomData;
/// #[derive(Copy, Clone)]
/// struct LeafDeserializeSeed {
///     mask_with: i64
/// }
/// impl<'de> DeserializeSeed<'de> for LeafDeserializeSeed {
///     type Value = i64;
/// 
///     fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
///         where D: serde::Deserializer<'de> 
///     {
///         Ok(self.mask_with ^ i64::deserialize(deserializer)?)
///     }
/// }
/// 
/// struct FooDeserializeSeed {
///     deserialize_a: LeafDeserializeSeed,
///     deserialize_b: LeafDeserializeSeed
/// };
/// 
/// impl_deserialize_seed_for_dependent_struct!{
///     struct Foo<'de> using FooDeserializeSeed {
///         a: i64: |seed: &FooDeserializeSeed| seed.deserialize_a,
///         b: i64: |seed: &FooDeserializeSeed| seed.deserialize_b
///     }
/// }
/// 
/// let mut deserializer = serde_json::Deserializer::new(serde_json::de::StrRead::new(r#"{
///     "a": 1,
///     "b": 0
/// }"#));
/// let deserialize_seed = FooDeserializeSeed {
///     deserialize_a: LeafDeserializeSeed { mask_with: 0 },
///     deserialize_b: LeafDeserializeSeed { mask_with: 1 }
/// };
/// let foo = deserialize_seed.deserialize(&mut deserializer).unwrap();
/// assert_eq!(1, foo.a);
/// assert_eq!(0 ^ 1, foo.b); // `b` should have been masked with `1` during deserialization
/// ```
/// 
/// Note that if `FooDeserializeSeed` should have generic parameters, these should be passed
/// in the following way:
/// ```
/// # use he_ring::impl_deserialize_seed_for_dependent_struct;
/// # use serde::*;
/// # use serde::de::DeserializeSeed;
/// # use std::marker::PhantomData;
/// struct FooDeserializeSeed<S>(S);
/// 
/// impl_deserialize_seed_for_dependent_struct!{
///     <{'de, S}> struct Foo<{'de, S}> using FooDeserializeSeed<S> {
///         a: S::Value: |seed: &FooDeserializeSeed<S>| seed.0.clone()
///     } where S: DeserializeSeed<'de> + Clone
/// }
/// ```
/// 
/// # But the lifetimes aren't exactly what they should be!?
/// 
/// Well, it depends on what you are trying to express. I implemented what I consider
/// the be the most powerful option, namely to allow `Foo` to borrow data from the
/// [`serde::Deserializer`], and thus depend on `'de`.
/// 
/// In the simpler (and possibly more common) case that `Foo` should own its data and
/// outlive the [`serde::Deserializer`], this causes a problem:
/// ```compile_fail
/// # use he_ring::impl_deserialize_seed_for_dependent_struct;
/// # use serde::*;
/// # use serde::de::DeserializeSeed;
/// # use std::marker::PhantomData;
/// struct FooDeserializeSeed;
/// 
/// impl_deserialize_seed_for_dependent_struct!{
///     struct Foo<'de> using FooDeserializeSeed {
///         a: String: |_| PhantomData::<String>
///     }
/// }
/// 
/// // compile error: `json_str` would have to have lifetime 'foo_lifetime
/// fn deserialize_foo_from_json<'foo_lifetime>(json_str: &str) -> Foo<'foo_lifetime> {
///     let mut deserializer = serde_json::Deserializer::new(serde_json::de::StrRead::new(json_str));
///     return FooDeserializeSeed.deserialize(&mut deserializer).unwrap();
/// }
/// ```
/// However, in these cases, it should suffice to manually convert `Foo` into some self-defined
/// struct `FooOwned` before returning it.
/// ```
/// # use he_ring::impl_deserialize_seed_for_dependent_struct;
/// # use serde::*;
/// # use serde::de::DeserializeSeed;
/// # use std::marker::PhantomData;
/// # struct FooDeserializeSeed;
/// # impl_deserialize_seed_for_dependent_struct!{
/// #     struct Foo<'de> using FooDeserializeSeed {
/// #         a: String: |_| PhantomData::<String>
/// #     }
/// # }
/// struct FooOwned {
///     a: String
/// }
/// fn deserialize_foo_from_json(json_str: &str) -> FooOwned {
///     let mut deserializer = serde_json::Deserializer::new(serde_json::de::StrRead::new(json_str));
///     let foo_borrowed = FooDeserializeSeed.deserialize(&mut deserializer).unwrap();
///     return FooOwned { a: foo_borrowed.a };
/// }
/// ```
/// 
#[macro_export]
macro_rules! impl_deserialize_seed_for_dependent_struct {
    (
        struct $deserialize_result_struct_name:ident<'de> using $deserialize_seed_type:ty {
            $($field:ident: $type:ty: $local_deserialize_seed:expr),*
        }
    ) => {
        impl_deserialize_seed_for_dependent_struct!{ <{'de,}> struct $deserialize_result_struct_name<{'de,}> using $deserialize_seed_type {
            $($field: $type: $local_deserialize_seed),*
        } where }
    };
    (
        <{'de, $($gen_args:tt)*}> struct $deserialize_result_struct_name:ident<{'de, $($deserialize_result_gen_args:tt)*}> using $deserialize_seed_type:ty {
            $($field:ident: $type:ty: $local_deserialize_seed:expr),*
        } where $($constraints:tt)*
    ) => {
        pub struct $deserialize_result_struct_name<'de, $($deserialize_result_gen_args)*> 
            where $($constraints)*
        {
            deserializer: std::marker::PhantomData<&'de ()>,
            $(pub $field: $type),*
        }
        impl<'de, $($gen_args)*> serde::de::DeserializeSeed<'de> for $deserialize_seed_type
            where $($constraints)*
        {
            type Value = $deserialize_result_struct_name<'de, $($deserialize_result_gen_args)*>;

            fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
                where D: serde::Deserializer<'de> 
            {
                use serde::de::*;

                type Field = Option<u32>;

                const fn get_const_len<const N: usize>(data: [&'static str; N]) -> usize {
                    N
                }
                const FIELD_COUNT: usize = get_const_len([$(stringify!($field)),*]);

                struct FieldVisitor;
                impl<'de> Visitor<'de> for FieldVisitor {

                    type Value = Field;

                    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                        std::fmt::Formatter::write_str(f, "field identifier")
                    }

                    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
                        where E: Error
                    {
                        if value >= FIELD_COUNT as u64 {
                            Ok(None)
                        } else {
                            Ok(Some(value as u32))
                        }
                    }

                    #[allow(unused_assignments)]
                    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
                        where E: Error
                    {
                        let mut current = 0;
                        $(
                            if value == stringify!($field) {
                                return Ok(Some(current));
                            }
                            current += 1;
                        )*
                        return Ok(None);
                    }

                    #[allow(unused_assignments)]
                    fn visit_bytes<E>(self, value: &[u8]) -> Result<Self::Value, E>
                        where E: Error
                    {
                        let mut current = 0;
                        $(
                            if value == stringify!($field).as_bytes() {
                                return Ok(Some(current));
                            }
                            current += 1;
                        )*
                        return Ok(None);
                    }
                }

                struct FieldDeserializer;
                impl<'de> DeserializeSeed<'de> for FieldDeserializer {
                    type Value = Field;

                    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
                        where D: serde::Deserializer<'de> 
                    {
                        deserializer.deserialize_identifier(FieldVisitor)
                    }
                }

                struct ResultVisitor<$($gen_args)*> {
                    deserialize_seed_base: $deserialize_seed_type
                }

                impl<'de, $($gen_args)*> Visitor<'de> for ResultVisitor<$($gen_args)*>
                    where $($constraints)*
                {
                    type Value = $deserialize_result_struct_name<'de, $($deserialize_result_gen_args)*>;

                    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                        std::fmt::Formatter::write_str(f, concat!("struct ", stringify!($deserialize_result_struct_name)))
                    }

                    #[allow(unused_assignments)]
                    fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                        where A: SeqAccess<'de>
                    {
                        let mut encountered_fields = 0;
                        Ok($deserialize_result_struct_name {
                            deserializer: std::marker::PhantomData,
                            $($field: {
                                let current_deserialize_seed = ($local_deserialize_seed)(&self.deserialize_seed_base);
                                let field_value = match seq.next_element_seed(current_deserialize_seed)? {
                                    Some(value) => value,
                                    None => return Err(Error::invalid_length(encountered_fields, &format!("struct {} with {} elements", stringify!($deserialize_result_struct_name), FIELD_COUNT).as_str()))
                                };
                                encountered_fields += 1;
                                field_value
                            }),*
                        })
                    }

                    #[allow(unused_assignments)]
                    fn visit_map<M>(self, mut map: M) -> Result<Self::Value, M::Error>
                        where M: MapAccess<'de>
                    {
                        $(
                            let mut $field: Option<$type> = None;
                        )*
                        while let Some(key) = map.next_key_seed(FieldDeserializer)? {
                            if let Some(key) = key {
                                let mut current = 0;
                                $(
                                    if key == current {
                                        if $field.is_some() {
                                            return Err(<M::Error as Error>::duplicate_field(stringify!($field)));
                                        }
                                        let current_deserialize_seed = ($local_deserialize_seed)(&self.deserialize_seed_base);
                                        $field = Some(map.next_value_seed(current_deserialize_seed)?);
                                    }
                                    current += 1;
                                )*
                            } else {
                                map.next_value::<IgnoredAny>()?;
                            }
                        }
                        $(
                            let $field: $type = match $field {
                                None => return Err(<M::Error as Error>::missing_field(stringify!($field))),
                                Some(value) => value
                            };
                        )*
                        return Ok($deserialize_result_struct_name { 
                            deserializer: PhantomData,
                            $($field),*
                        });
                    }
                }

                return deserializer.deserialize_struct(
                    stringify!($deserialize_result_struct_name),
                    &[$(stringify!($field)),*],
                    ResultVisitor { deserialize_seed_base: self }
                )
            }
        }
    };
}

pub struct DeserializeSeedDependentTuple<'de, T0, F, T1>
    where T0: DeserializeSeed<'de>,
        T1: DeserializeSeed<'de>,
        F: FnOnce(&T0::Value) -> T1
{
    deserializer: PhantomData<&'de ()>,
    first: T0,
    derive_second: F
}

impl<'de, T0, F, T1> DeserializeSeedDependentTuple<'de, T0, F, T1>
    where T0: DeserializeSeed<'de>,
        T1: DeserializeSeed<'de>,
        F: FnOnce(&T0::Value) -> T1
{
    pub fn new(first: T0, derive_second: F) -> Self {
        Self {
            deserializer: PhantomData,
            first: first,
            derive_second: derive_second
        }
    }
}

impl<'de, T0, F, T1> DeserializeSeed<'de> for DeserializeSeedDependentTuple<'de, T0, F, T1>
    where T0: DeserializeSeed<'de>,
        T1: DeserializeSeed<'de>,
        F: FnOnce(&T0::Value) -> T1
{
    type Value = (T0::Value, T1::Value);

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where D: Deserializer<'de>
    {
        pub struct ResultVisitor<'de, T0, F, T1>
            where T0: DeserializeSeed<'de>,
                T1: DeserializeSeed<'de>,
                F: FnOnce(&T0::Value) -> T1
        {
            deserializer: PhantomData<&'de ()>,
            first: T0,
            derive_second: F
        }

        impl<'de, T0, F, T1> Visitor<'de> for ResultVisitor<'de, T0, F, T1>
            where T0: DeserializeSeed<'de>,
                T1: DeserializeSeed<'de>,
                F: FnOnce(&T0::Value) -> T1
        {
            type Value = (T0::Value, T1::Value);

            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "a tuple with 2 elements")
            }

            fn visit_seq<A>(self, mut seq: A) -> Result<Self::Value, A::Error>
                where A: SeqAccess<'de>
            {
                if let Some(first) = seq.next_element_seed(self.first)? {
                    if let Some(second) = seq.next_element_seed((self.derive_second)(&first))? {
                        if let Some(_) = seq.next_element::<IgnoredAny>()? {
                            return Err(<A::Error as serde::de::Error>::invalid_length(3, &"a tuple with 2 elements"));
                        } else {
                            return Ok((first, second));
                        }
                    } else {
                        return Err(<A::Error as serde::de::Error>::invalid_length(1, &"a tuple with 2 elements"));
                    }
                } else {
                    return Err(<A::Error as serde::de::Error>::invalid_length(0, &"a tuple with 2 elements"));
                }
            }
        }

        return deserializer.deserialize_tuple(2, ResultVisitor {
            deserializer: PhantomData,
            first: self.first,
            derive_second: self.derive_second
        });
    }
}

const fn get_const_len<const N: usize>(data: [&'static str; N]) -> usize {
    N
}

macro_rules! deserialize_seed_tuple {
    ($name:ident, $(( $gen_arg:ident, $field:ident)),*) => {
        pub struct $name<'de, $($gen_arg),*>
            where $($gen_arg: DeserializeSeed<'de>),*
        {
            deserializer: PhantomData<&'de ()>,
            $($field: $gen_arg),*
        }

        impl<'de, $($gen_arg),*> DeserializeSeed<'de> for $name<'de, $($gen_arg),*>
            where $($gen_arg: DeserializeSeed<'de>),*
        {
            type Value = ($(<$gen_arg as DeserializeSeed<'de>>::Value),*,);

            fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
                where D: serde::Deserializer<'de>
            {
                struct ResultVisitor<'de, $($gen_arg),*>
                    where $($gen_arg: DeserializeSeed<'de>),*
                {
                    deserializer: PhantomData<&'de ()>,
                    $($field: $gen_arg),*
                }

                impl<'de, $($gen_arg),*> Visitor<'de> for ResultVisitor<'de, $($gen_arg),*>
                    where $($gen_arg: DeserializeSeed<'de>),*
                {
                    type Value = ($(<$gen_arg as DeserializeSeed<'de>>::Value),*,);

                    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                        write!(f, "a tuple with {} elements", get_const_len([$(stringify!($gen_arg)),*]))
                    }

                    #[allow(unused_assignments)]
                    fn visit_seq<Seq>(self, mut seq: Seq) -> Result<Self::Value, Seq::Error>
                        where Seq: SeqAccess<'de>
                    {
                        const TUPLE_LEN: usize = get_const_len([$(stringify!($gen_arg)),*]);
                        let mut current = 0;
                        Ok(($(
                            {
                                let result = if let Some(result) = seq.next_element_seed(self.$field)? {
                                    result
                                } else {
                                    return Err(<Seq::Error as serde::de::Error>::invalid_length(current, &format!("a tuple with {} elements", TUPLE_LEN).as_str()));
                                };
                                current += 1;
                                result
                            }
                        ),*,))
                    }
                }

                return deserializer.deserialize_tuple(get_const_len([$(stringify!($gen_arg)),*]), ResultVisitor {
                    deserializer: PhantomData,
                    $($field: self.$field),*
                });
            }
        }
    };
}

deserialize_seed_tuple!{ DeserializeSeedTuple1, (T0, t0) }
deserialize_seed_tuple!{ DeserializeSeedTuple2, (T0, t0), (T1, t1) }
deserialize_seed_tuple!{ DeserializeSeedTuple3, (T0, t0), (T1, t1), (T2, t2) }
deserialize_seed_tuple!{ DeserializeSeedTuple4, (T0, t0), (T1, t1), (T2, t2), (T3, t3) }
deserialize_seed_tuple!{ DeserializeSeedTuple5, (T0, t0), (T1, t1), (T2, t2), (T3, t3), (T4, t4) }
