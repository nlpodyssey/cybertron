// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package sliceutils provides types and functions for various operations over
// sliceutils of different types.
package sliceutils

// Ordered is a type constraint that permits any ordered type, that is, any
// type supporting the operators < <= >= >.
type Ordered interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64 |
		~string
}
