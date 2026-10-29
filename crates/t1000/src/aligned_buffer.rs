// SPDX-License-Identifier: MIT OR Apache-2.0
// Copyright (c) 2026 Terrence A. Jones SR. All rights reserved.

//! 64-Byte Cache-Line Aligned Buffer Subsystem for AVX2 and AVX-512 SIMD Vectorization.
//!
//! Provides [`AlignedBuffer`], a contiguous heap-allocated array guaranteed to be aligned
//! to 64-byte boundaries (`addr % 64 == 0`). This eliminates split-cache-line stalls
//! and enables aligned vector operations (`_mm256_load_pd`, `_mm512_load_pd`, `_mm256_stream_pd`).

use serde::de::{SeqAccess, Visitor};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::alloc::{alloc_zeroed, dealloc, realloc, Layout};
use std::fmt;
use std::ops::{Deref, DerefMut, Index, IndexMut};
use std::ptr::NonNull;

/// Cache line alignment for modern x86_64 architectures (64 bytes).
pub const CACHE_LINE_ALIGN: usize = 64;

/// A contiguous 64-byte cache-line aligned buffer of 64-bit floating point numbers (`f64`).
pub struct AlignedBuffer {
    ptr: NonNull<f64>,
    len: usize,
    capacity: usize,
}

impl AlignedBuffer {
    /// Constructs an empty aligned buffer with zero capacity.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            capacity: 0,
        }
    }

    /// Allocates an aligned buffer with `len` elements initialized to `fill_val`.
    pub fn new(len: usize, fill_val: f64) -> Self {
        if len == 0 {
            return Self::empty();
        }
        let mut buf = Self::zeroed(len);
        buf.as_mut_slice().fill(fill_val);
        buf
    }

    /// Allocates an aligned buffer with `len` elements initialized to `0.0`.
    pub fn zeroed(len: usize) -> Self {
        if len == 0 {
            return Self::empty();
        }
        let capacity = len.max(8);
        let bytes = capacity
            .checked_mul(std::mem::size_of::<f64>())
            .expect("capacity overflow");
        let layout = Layout::from_size_align(bytes, CACHE_LINE_ALIGN).expect("valid layout");

        let raw = unsafe { alloc_zeroed(layout) as *mut f64 };
        let ptr = NonNull::new(raw).expect("aligned memory allocation failed");

        Self { ptr, len, capacity }
    }

    /// Creates an aligned buffer by copying data from a slice.
    pub fn from_slice(slice: &[f64]) -> Self {
        let mut buf = Self::zeroed(slice.len());
        buf.as_mut_slice().copy_from_slice(slice);
        buf
    }

    /// Creates an aligned buffer with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::empty();
        }
        let cap = capacity.max(8);
        let bytes = cap
            .checked_mul(std::mem::size_of::<f64>())
            .expect("capacity overflow");
        let layout = Layout::from_size_align(bytes, CACHE_LINE_ALIGN).expect("valid layout");

        let raw = unsafe { alloc_zeroed(layout) as *mut f64 };
        let ptr = NonNull::new(raw).expect("aligned memory allocation failed");

        Self {
            ptr,
            len: 0,
            capacity: cap,
        }
    }

    /// Appends an element to the back of the buffer.
    pub fn push(&mut self, val: f64) {
        if self.len == self.capacity {
            let new_cap = if self.capacity == 0 { 8 } else { self.capacity * 2 };
            self.reserve_capacity(new_cap);
        }
        unsafe {
            self.ptr.as_ptr().add(self.len).write(val);
        }
        self.len += 1;
    }

    fn reserve_capacity(&mut self, new_cap: usize) {
        let old_bytes = self.capacity * std::mem::size_of::<f64>();
        let new_bytes = new_cap * std::mem::size_of::<f64>();
        let old_layout = Layout::from_size_align(old_bytes, CACHE_LINE_ALIGN).expect("valid layout");

        let raw = if self.capacity == 0 {
            let layout = Layout::from_size_align(new_bytes, CACHE_LINE_ALIGN).expect("valid layout");
            unsafe { alloc_zeroed(layout) as *mut f64 }
        } else {
            unsafe { realloc(self.ptr.as_ptr() as *mut u8, old_layout, new_bytes) as *mut f64 }
        };

        self.ptr = NonNull::new(raw).expect("reallocation failed");
        self.capacity = new_cap;
    }

    /// Returns `true` if the underlying buffer pointer is 64-byte cache-line aligned.
    #[inline(always)]
    pub fn is_aligned_64(&self) -> bool {
        (self.ptr.as_ptr() as usize).is_multiple_of(CACHE_LINE_ALIGN)
    }

    /// Returns a shared slice of the buffer elements.
    #[inline(always)]
    pub fn as_slice(&self) -> &[f64] {
        if self.len == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
        }
    }

    /// Returns a mutable slice of the buffer elements.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f64] {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
        }
    }
}

impl Drop for AlignedBuffer {
    fn drop(&mut self) {
        if self.capacity > 0 {
            let bytes = self.capacity * std::mem::size_of::<f64>();
            let layout = Layout::from_size_align(bytes, CACHE_LINE_ALIGN).expect("valid layout");
            unsafe {
                dealloc(self.ptr.as_ptr() as *mut u8, layout);
            }
        }
    }
}

impl Clone for AlignedBuffer {
    fn clone(&self) -> Self {
        Self::from_slice(self.as_slice())
    }
}

impl Default for AlignedBuffer {
    fn default() -> Self {
        Self::empty()
    }
}

impl Deref for AlignedBuffer {
    type Target = [f64];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for AlignedBuffer {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl<I> Index<I> for AlignedBuffer
where
    I: std::slice::SliceIndex<[f64]>,
{
    type Output = I::Output;
    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        Index::index(self.as_slice(), index)
    }
}

impl<I> IndexMut<I> for AlignedBuffer
where
    I: std::slice::SliceIndex<[f64]>,
{
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        IndexMut::index_mut(self.as_mut_slice(), index)
    }
}

impl From<Vec<f64>> for AlignedBuffer {
    fn from(vec: Vec<f64>) -> Self {
        Self::from_slice(&vec)
    }
}

impl From<&[f64]> for AlignedBuffer {
    fn from(slice: &[f64]) -> Self {
        Self::from_slice(slice)
    }
}

impl FromIterator<f64> for AlignedBuffer {
    fn from_iter<T: IntoIterator<Item = f64>>(iter: T) -> Self {
        let v: Vec<f64> = iter.into_iter().collect();
        Self::from(v)
    }
}

impl<'a> IntoIterator for &'a AlignedBuffer {
    type Item = &'a f64;
    type IntoIter = std::slice::Iter<'a, f64>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<'a> IntoIterator for &'a mut AlignedBuffer {
    type Item = &'a mut f64;
    type IntoIter = std::slice::IterMut<'a, f64>;
    fn into_iter(self) -> Self::IntoIter {
        self.as_mut_slice().iter_mut()
    }
}

impl PartialEq for AlignedBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for AlignedBuffer {}

impl fmt::Debug for AlignedBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AlignedBuffer({:?})", self.as_slice())
    }
}

unsafe impl Send for AlignedBuffer {}
unsafe impl Sync for AlignedBuffer {}

impl Serialize for AlignedBuffer {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.collect_seq(self.as_slice())
    }
}

impl<'de> Deserialize<'de> for AlignedBuffer {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct AlignedBufferVisitor;

        impl<'de> Visitor<'de> for AlignedBufferVisitor {
            type Value = AlignedBuffer;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("a sequence of f64 amplitudes")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
                let mut buf = AlignedBuffer::with_capacity(seq.size_hint().unwrap_or(0));
                while let Some(elem) = seq.next_element()? {
                    buf.push(elem);
                }
                Ok(buf)
            }
        }

        deserializer.deserialize_seq(AlignedBufferVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aligned_buffer_64_byte_alignment() {
        for len in [1, 2, 4, 8, 16, 64, 1024, 65536] {
            let buf = AlignedBuffer::zeroed(len);
            assert_eq!(buf.len(), len);
            assert!(buf.is_aligned_64());
            assert_eq!((buf.as_ptr() as usize) % 64, 0);
        }
    }

    #[test]
    fn test_aligned_buffer_clone_and_mutation() {
        let mut buf = AlignedBuffer::new(16, 1.5);
        assert!(buf.is_aligned_64());
        buf[0] = 3.0;
        buf[15] = 4.5;

        let cloned = buf.clone();
        assert!(cloned.is_aligned_64());
        assert_eq!(cloned[0], 3.0);
        assert_eq!(cloned[1], 1.5);
        assert_eq!(cloned[15], 4.5);
    }

    #[test]
    fn test_aligned_buffer_serde_roundtrip() {
        let buf = AlignedBuffer::from_slice(&[1.0, 2.0, 3.0, 4.0]);
        let json = serde_json::to_string(&buf).expect("serialize");
        let restored: AlignedBuffer = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(buf, restored);
        assert!(restored.is_aligned_64());
    }
}
