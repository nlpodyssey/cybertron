// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sliceutils

// OrderedHeap is a min-heap of Ordered values.
type OrderedHeap[T Ordered] []T

// Len returns the length of the heap.
func (t OrderedHeap[_]) Len() int {
	return len(t)
}

// Less reports whether the value at index i is less than the value at index j.
func (t OrderedHeap[_]) Less(i, j int) bool {
	return t[i] < t[j]
}

// Swap swaps the elements at indices i and j.
func (t OrderedHeap[_]) Swap(i, j int) {
	t[i], t[j] = t[j], t[i]
}

// Push appends the value x to the heap.
func (t *OrderedHeap[T]) Push(x any) {
	*t = append(*t, x.(T))
}

// Pop removes the last element from the heap and returns its value.
func (t *OrderedHeap[T]) Pop() any {
	lastIndex := len(*t) - 1
	x := (*t)[lastIndex]
	*t = (*t)[:lastIndex]
	return x
}
