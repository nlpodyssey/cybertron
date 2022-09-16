// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensim

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/numpy"
	"github.com/nlpodyssey/gopickle/types"
)

type KeyedVectorsClass struct{}

type Word2VecKeyedVectorsClass = KeyedVectorsClass // For back compatibility.

type KeyedVectors struct {
	Vocab      map[string]*Vocab
	VectorSize int
	Index2Word []string
	Vectors    *numpy.NDArray

	NumPys             *types.List
	SciPys             *types.List
	Ignoreds           *types.List
	RecursiveSaveloads *types.List
}

func (KeyedVectorsClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("KeyedVectorsClass: want no arguments, got %d: %#v", len(args), args)
	}
	return &KeyedVectors{}, nil
}

func (kv *KeyedVectors) PyDictSet(k, v any) (err error) {
	switch k {
	case "index2word":
		var l *types.List
		err = conversion.AssignAssertedType(v, &l)
		if err == nil {
			err = conversion.AssignListToSlice(l, &kv.Index2Word)
		}
	case "vector_size":
		err = conversion.AssignAssertedType(v, &kv.VectorSize)
	case "vectors":
		err = conversion.AssignAssertedType(v, &kv.Vectors)
	case "vectors_norm":
		if v != nil {
			err = fmt.Errorf("only nil is supported, got %T: %#v", v, v)
		}
	case "vocab":
		var d *types.Dict
		err = conversion.AssignAssertedType(v, &d)
		if err == nil {
			err = conversion.AssignDictToMap(d, &kv.Vocab)
		}
	case "__ignoreds":
		err = conversion.AssignAssertedType(v, &kv.Ignoreds)
	case "__numpys":
		err = conversion.AssignAssertedType(v, &kv.NumPys)
	case "__recursive_saveloads":
		err = conversion.AssignAssertedType(v, &kv.RecursiveSaveloads)
	case "__scipys":
		err = conversion.AssignAssertedType(v, &kv.SciPys)
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("KeyedVectors: dict key %q: %w", k, err)
	}
	return err
}
