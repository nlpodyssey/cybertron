// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ptr

// Of allocates a new variable of a given value and returns a pointer to it.
func Of[T any](value T) *T {
	return &value
}
