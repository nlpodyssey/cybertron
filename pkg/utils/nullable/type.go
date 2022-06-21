// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package nullable

// Type can contain a nil value.
type Type[T any] struct {
	// Value is the value (can be nil)
	Value T
	// Valid is true if the value is not nil.
	Valid bool
}

// Any creates a new Type with the given value.
func Any[T any](v *T) Type[T] {
	if v == nil {
		return Type[T]{}
	}
	return Type[T]{Value: *v, Valid: true}
}

// Int creates a new NullableType with the given int value.
func Int[T int | int64](v *T) Type[int] {
	if v == nil {
		return Type[int]{}
	}
	return Type[int]{Value: int(*v), Valid: true}
}
