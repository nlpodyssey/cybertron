// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sliceutils

import "container/heap"

// ReverseHeap returns the reverse order for data.
func ReverseHeap(data heap.Interface) heap.Interface {
	return &reverseHeap{data}
}

type reverseHeap struct {
	// This embedded heap.Interface permits ReverseHeap to use the methods of
	// another heap.Interface implementations.
	heap.Interface
}

// Less returns the opposite of the embedded implementations's Less method.
func (r reverseHeap) Less(i, j int) bool {
	return r.Interface.Less(j, i)
}
