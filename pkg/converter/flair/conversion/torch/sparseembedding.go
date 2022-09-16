// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
	"github.com/nlpodyssey/spago/mat"
)

type SparseEmbeddingClass struct{}

type SparseEmbedding struct {
	Module
	NumEmbeddings   int
	EmbeddingDim    int
	PaddingIdx      *int
	MaxNorm         *float64
	NormType        *float64
	ScaleGradByFreq *bool
	Sparse          *bool
	Weight          []mat.Matrix
}

func (SparseEmbeddingClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("SparseEmbeddingClass: unsupported arguments: %#v", args)
	}
	return &SparseEmbedding{}, nil
}

func (se *SparseEmbedding) PyDictSet(k, v any) (err error) {
	if err := se.Module.PyDictSet(k, v); err == nil {
		return nil
	} else if !IsUnexpectedModuleDictKey(err) {
		return fmt.Errorf("SparseEmbedding: %w", err)
	}

	switch k {
	case "num_embeddings":
		err = conversion.AssignAssertedType(v, &se.NumEmbeddings)
	case "embedding_dim":
		err = conversion.AssignAssertedType(v, &se.EmbeddingDim)
	case "padding_idx":
		err = conversion.AssignOptionalAssertedType(v, &se.PaddingIdx)
	case "max_norm":
		err = conversion.AssignOptionalAssertedType(v, &se.MaxNorm)
	case "norm_type":
		// err = conversion.AssignOptionalAssertedType(v, &se.NormType) // TODO: int to float conversion
	case "scale_grad_by_freq":
		err = conversion.AssignOptionalAssertedType(v, &se.ScaleGradByFreq)
	case "sparse":
		err = conversion.AssignOptionalAssertedType(v, &se.Sparse)
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("SparseEmbedding: dict key %q: %w", k, err)
	}
	return err
}

func (se *SparseEmbedding) LoadStateDictEntry(k string, v any) (err error) {
	switch k {
	case "weight":
		err = se.setWeight(v)
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("SparseEmbedding: state dict key %q: %w", k, err)
	}
	return err
}

func (se *SparseEmbedding) setWeight(v any) error {
	t, err := AnyToTensor(v, []int{se.NumEmbeddings, se.EmbeddingDim})
	if err == nil {
		se.Weight, err = conversion.Tensor2DToSliceOfVectors(t)
	}
	return err
}
