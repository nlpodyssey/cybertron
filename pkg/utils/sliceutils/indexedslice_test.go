// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sliceutils

import (
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var _ sort.Interface = IndexedSlice[int]{}

func TestNewIndexedSlice(t *testing.T) {
	values := []int{7, 8, 9}
	s := NewIndexedSlice(values)
	assert.Equal(t, values, s.Slice)
	assert.Equal(t, []int{0, 1, 2}, s.Indices)
}

func TestIndexedSlice_Len(t *testing.T) {
	tests := []struct {
		s IndexedSlice[int]
		l int
	}{
		{NewIndexedSlice([]int{}), 0},
		{NewIndexedSlice([]int{42}), 1},
		{NewIndexedSlice([]int{8, 9}), 2},
		{NewIndexedSlice([]int{1, 3, 5}), 3},
	}
	for _, tt := range tests {
		assert.Equalf(t, tt.l, tt.s.Len(), "len of %v", tt.s)
	}
}

func TestIndexedSlice_Less(t *testing.T) {
	s := NewIndexedSlice([]int{5, 1, 9, 1})
	tests := [4][4]bool{
		{false, false, true, false},
		{true, false, true, false},
		{false, false, false, false},
		{true, false, true, false},
	}
	for i, iv := range tests {
		for j, want := range iv {
			assert.Equalf(t, want, s.Less(i, j), "Less(%d, %d)", i, j)
		}
	}
}

func TestIndexedSlice_Swap(t *testing.T) {
	s := NewIndexedSlice([]int{7, 8, 9})
	swaps := []struct {
		i, j       int
		wantSlice  []int
		wantIndies []int
	}{
		{0, 1, []int{8, 7, 9}, []int{1, 0, 2}},
		{0, 2, []int{9, 7, 8}, []int{2, 0, 1}},
		{1, 2, []int{9, 8, 7}, []int{2, 1, 0}},
		{2, 0, []int{7, 8, 9}, []int{0, 1, 2}},
		{1, 1, []int{7, 8, 9}, []int{0, 1, 2}},
	}
	for _, sw := range swaps {
		s.Swap(sw.i, sw.j)
		require.EqualValuesf(t, sw.wantSlice, s.Slice, "slice after Swap(%d, %d)", sw.i, sw.j)
		require.EqualValuesf(t, sw.wantIndies, s.Indices, "indices after Swap(%d, %d)", sw.i, sw.j)
	}
}
