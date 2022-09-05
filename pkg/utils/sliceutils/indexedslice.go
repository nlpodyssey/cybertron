// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sliceutils

// IndexedSlice allows sorting a slice of Ordered values without losing
// track of the initial (pre-sorting) index of each element.
type IndexedSlice[T Ordered] struct {
	// Slice of values exposed to the sorting operation.
	Slice []T
	// Indices is initialized with the index of each element
	// in the original slice, and is sorted in parallel with Slice.
	Indices []int
}

// NewIndexedSlice creates a new IndexedSlice.
func NewIndexedSlice[T Ordered](slice []T) IndexedSlice[T] {
	indices := make([]int, len(slice))
	for i := range indices {
		indices[i] = i
	}
	s := IndexedSlice[T]{
		Slice:   slice,
		Indices: indices,
	}
	return s
}

// Len returns the length of the slice.
func (s IndexedSlice[_]) Len() int {
	return len(s.Indices)
}

// Less reports whether the value at index i is less than the value at index j.
func (s IndexedSlice[T]) Less(i, j int) bool {
	return s.Slice[i] < s.Slice[j]
}

// Swap swaps the elements at indices i and j, on both Indices and Slice.
func (s IndexedSlice[T]) Swap(i, j int) {
	in := s.Indices
	sl := s.Slice
	in[i], in[j] = in[j], in[i]
	sl[i], sl[j] = sl[j], sl[i]
}
