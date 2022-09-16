// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package numpy

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
	"github.com/nlpodyssey/gopickle/types"
	"github.com/nlpodyssey/spago/mat"
)

type NDArrayClass struct{}

type NDArray struct {
	Shape     []int
	DTypeStr  string
	DType     *DType
	IsFortran bool
	RawData   []byte
}

func (n NDArrayClass) PyNew(args ...any) (any, error) {
	if len(args) == 0 || len(args) > 2 {
		return nil, fmt.Errorf("NDArrayClass: want 1 or 2 arguments, got %d: %#v", len(args), args)
	}

	nda := new(NDArray)

	shape, err := conversion.AssertType[*types.Tuple](args[0])
	if err == nil {
		err = conversion.AssignTupleToSlice(shape, &nda.Shape)
	}
	if err != nil {
		return nil, fmt.Errorf("NDArrayClass: 1st arg (shape): %w", err)
	}

	if len(args) == 1 {
		return nda, nil
	}

	err = conversion.AssignAssertedType(args[1], &nda.DTypeStr)
	if err != nil {
		return nil, fmt.Errorf("NDArrayClass: 2nd arg (dtype): %w", err)
	}

	return nda, nil
}

func (n *NDArray) PySetState(state any) error {
	st, err := conversion.AssertType[*types.Tuple](state)
	if err != nil {
		return fmt.Errorf("NDArray: failed to set state: %w", err)
	}
	if st.Len() != 5 {
		return fmt.Errorf("NDArray: want state Tuple of 5 items, got length %d: %#v", st.Len(), *st)
	}

	_, err = conversion.AssertType[int](st.Get(0))
	if err != nil {
		return fmt.Errorf("NDArray: 1st state Tuple item (version): %w", err)
	}

	shapeTuple, err := conversion.AssertType[*types.Tuple](st.Get(1))
	if err == nil {
		err = conversion.AssignTupleToSlice(shapeTuple, &n.Shape)
	}
	if err != nil {
		return fmt.Errorf("NDArray: 2ns state Tuple item (shape): %w", err)
	}

	err = conversion.AssignAssertedType(st.Get(2), &n.DType)
	if err != nil {
		return fmt.Errorf("NDArray: 3rd state Tuple item (dtype): %w", err)
	}
	n.DTypeStr = ""

	err = conversion.AssignAssertedType(st.Get(3), &n.IsFortran)
	if err != nil {
		return fmt.Errorf("NDArray: 4th state Tuple item (isFortran): %w", err)
	}

	err = conversion.AssignAssertedType(st.Get(4), &n.RawData)
	if err != nil {
		return fmt.Errorf("NDArray: 5th state Tuple item (rawdata): %w", err)
	}

	return nil
}

func (n *NDArray) SliceOfVectors() ([]mat.Matrix, error) {
	switch {
	case len(n.Shape) != 2:
		return nil, fmt.Errorf("want 2 dimensions, got shape %v", n.Shape)
	case n.DType == nil:
		return nil, fmt.Errorf("NDArray.DType is not present")
	case n.DType.ByteOrder != "<":
		return nil, fmt.Errorf(`invalid or unimplemented byte order: want little-endian ("<"), got %q`, n.DType.ByteOrder)
	case n.DType.DType != "f4":
		return nil, fmt.Errorf(`invalid or unimplemented dtype: want "f4", got %q`, n.DType.DType)
	case len(n.RawData)%4 != 0 || len(n.RawData)/4 != n.Shape[0]*n.Shape[1]:
		return nil, fmt.Errorf("raw data length %d does not match shape %v", len(n.RawData), n.Shape)
	}

	vectors := make([]mat.Matrix, n.Shape[0])
	buf := make([]float32, n.Shape[1])
	r := bytes.NewReader(n.RawData)

	for i := range vectors {
		err := binary.Read(r, binary.LittleEndian, buf)
		if err != nil {
			return nil, fmt.Errorf("failed to read raw data for vector at index %d: %w", i, err)
		}
		vectors[i] = mat.NewVecDense[float32](buf)
	}

	if _, e := r.ReadByte(); e != io.EOF {
		return nil, fmt.Errorf("want EOF after reading all vectors, got %v", e)
	}

	return vectors, nil
}
