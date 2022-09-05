// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sliceutils

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestReverseHeap_Less(t *testing.T) {
	h := ReverseHeap(&dummyHeap{5, 1, 9, 1})
	tests := [4][4]bool{
		{false, true, false, true},
		{false, false, false, false},
		{true, true, false, true},
		{false, false, false, false},
	}
	for i, iv := range tests {
		for j, want := range iv {
			assert.Equalf(t, want, h.Less(i, j), "Less(%d, %d)", i, j)
		}
	}
}

type dummyHeap []int

func (h dummyHeap) Less(i, j int) bool { return h[i] < h[j] }
func (h dummyHeap) Len() int           { panic("unexpected call to Len") }
func (h dummyHeap) Swap(i, j int)      { panic("unexpected call to Swap") }
func (h dummyHeap) Push(x any)         { panic("unexpected call to Push") }
func (h dummyHeap) Pop() any           { panic("unexpected call to Pop") }
