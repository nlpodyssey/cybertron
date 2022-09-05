// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sliceutils

import (
	"container/heap"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var _ heap.Interface = new(OrderedHeap[int])

func TestOrderedHeap_Len(t *testing.T) {
	tests := []struct {
		h *OrderedHeap[int]
		l int
	}{
		{&OrderedHeap[int]{}, 0},
		{&OrderedHeap[int]{42}, 1},
		{&OrderedHeap[int]{8, 9}, 2},
		{&OrderedHeap[int]{1, 3, 5}, 3},
	}
	for _, tt := range tests {
		assert.Equalf(t, tt.l, tt.h.Len(), "len of %v", *tt.h)
	}
}

func TestOrderedHeap_Less(t *testing.T) {
	h := &OrderedHeap[int]{5, 1, 9, 1}
	tests := [4][4]bool{
		{false, false, true, false},
		{true, false, true, false},
		{false, false, false, false},
		{true, false, true, false},
	}
	for i, iv := range tests {
		for j, want := range iv {
			assert.Equalf(t, want, h.Less(i, j), "Less(%d, %d)", i, j)
		}
	}
}

func TestOrderedHeap_Swap(t *testing.T) {
	h := &OrderedHeap[int]{0, 1, 2}
	swaps := []struct {
		i, j int
		want []int
	}{
		{0, 1, []int{1, 0, 2}},
		{0, 2, []int{2, 0, 1}},
		{1, 2, []int{2, 1, 0}},
		{2, 0, []int{0, 1, 2}},
		{1, 1, []int{0, 1, 2}},
	}
	for _, s := range swaps {
		h.Swap(s.i, s.j)
		require.EqualValuesf(t, s.want, *h, "after Swap(%d, %d)", s.i, s.j)
	}
}

func TestOrderedHeap_Push(t *testing.T) {
	h := &OrderedHeap[int]{}
	pushes := []struct {
		x    int
		want []int
	}{
		{1, []int{1}},
		{3, []int{1, 3}},
		{5, []int{1, 3, 5}},
		{7, []int{1, 3, 5, 7}},
	}
	for _, p := range pushes {
		h.Push(p.x)
		require.EqualValuesf(t, p.want, *h, "after Push(%d)", p.x)
	}
}

func TestOrderedHeap_Pop(t *testing.T) {
	h := &OrderedHeap[int]{1, 3, 5}

	pops := []struct {
		x    int
		rest []int
	}{
		{5, []int{1, 3}},
		{3, []int{1}},
		{1, []int{}},
	}
	for i, p := range pops {
		x := h.Pop()
		assert.Equalf(t, p.x, x, "value after pop #%d", i+1)
		require.EqualValuesf(t, p.rest, *h, "remaining after pop #%d", i+1)
	}
}
