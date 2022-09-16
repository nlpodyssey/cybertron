// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conversion

import (
	"fmt"

	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
	"github.com/nlpodyssey/spago/mat"
)

func AssertType[T any](v any) (vt T, err error) {
	var ok bool
	vt, ok = v.(T)
	if !ok {
		return vt, fmt.Errorf("failed to assert type %T, got type %T: %#v", vt, v, v)
	}
	return vt, nil
}

func AssignAssertedType[T any](v any, to *T) (err error) {
	*to, err = AssertType[T](v)
	return err
}

func AssignOptionalAssertedType[T any](v any, to **T) (err error) {
	if v == nil {
		*to = nil
		return nil
	}
	*to = new(T)
	**to, err = AssertType[T](v)
	return err
}

func DictToMap[K comparable, V any](d *types.Dict) (map[K]V, error) {
	m := make(map[K]V, d.Len())
	for _, e := range *d {
		k, err := AssertType[K](e.Key)
		if err != nil {
			return nil, fmt.Errorf("failed to convert *Dict to %T: invalid key type: %w", m, err)
		}
		v, err := AssertType[V](e.Value)
		if err != nil {
			return nil, fmt.Errorf("failed to convert *Dict to %T: invalid value type: %w", m, err)
		}
		m[k] = v
	}
	return m, nil
}

func AssignDictToMap[K comparable, V any](d *types.Dict, m *map[K]V) (err error) {
	*m, err = DictToMap[K, V](d)
	return err
}

func ListToSlice[T any](l *types.List) ([]T, error) {
	s := make([]T, l.Len())
	for i, pv := range *l {
		v, err := AssertType[T](pv)
		if err != nil {
			return nil, fmt.Errorf("failed to convert *List to %T: invalid value type: %w", s, err)
		}
		s[i] = v
	}
	return s, nil
}

func AssignListToSlice[T any](l *types.List, s *[]T) (err error) {
	*s, err = ListToSlice[T](l)
	return err
}

func TupleToSlice[T any](t *types.Tuple) ([]T, error) {
	s := make([]T, t.Len())
	for i, pv := range *t {
		v, err := AssertType[T](pv)
		if err != nil {
			return nil, fmt.Errorf("failed to convert *Tuple to %T: invalid value type: %w", s, err)
		}
		s[i] = v
	}
	return s, nil
}

func AssignTupleToSlice[T any](t *types.Tuple, s *[]T) (err error) {
	*s, err = TupleToSlice[T](t)
	return err
}

// GetTensorData returns the data of a PyTorch tensor as a mat.Float slice.
// It returns the data using the row-major representation, possibly converting column-major order to row-major order.
func GetTensorData(t *pytorch.Tensor) ([]float32, error) {
	if len(t.Size) == 0 || len(t.Size) > 2 {
		return nil, fmt.Errorf("failed to convert tensor: want 1 or 2 dimensions, got %d", len(t.Size))
	}

	size := t.Size[0]
	if len(t.Size) > 1 {
		size *= t.Size[1]
	}

	source, ok := t.Source.(*pytorch.FloatStorage)
	if !ok {
		return nil, fmt.Errorf("invalid or unimplemented torch source storage type %T", t.Source)
	}

	orig := source.Data[t.StorageOffset : t.StorageOffset+size]
	data := make([]float32, len(orig))

	if len(t.Size) == 1 || t.Size[0] == 1 || t.Size[1] == 1 || t.Stride[1] == 1 {
		copy(data, orig)
		return data, nil
	}

	s0, s1 := t.Size[1], t.Size[0]
	for i := 0; i < s0; i++ {
		for j := 0; j < s1; j++ {
			data[i+j*s0] = orig[j+i*s1]
		}
	}

	return data, nil
}

func Tensor2DToSliceOfVectors(t *pytorch.Tensor) ([]mat.Matrix, error) {
	if len(t.Size) != 2 {
		return nil, fmt.Errorf("want 2 dimensions, got size %v", t.Size)
	}
	data, err := GetTensorData(t)
	if err != nil {
		return nil, err
	}

	vectors := make([]mat.Matrix, t.Size[0])
	vecSize := t.Size[1]

	for i := range vectors {
		from := vecSize * i
		vectors[i] = mat.NewVecDense[float32](data[from : from+vecSize])
	}

	return vectors, nil
}
