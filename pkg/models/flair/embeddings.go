// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/linear"
)

type Embeddings struct {
	nn.Module
	TokensEncoder []TokensEncoder
	Projection    *linear.Model
}

type TokensEncoder interface {
	nn.Model
	EncodeTokens(tokens []string) []mat.Tensor
}

func (m *Embeddings) EncodeTokens(tokens []string) []mat.Tensor {
	encoded := make([][]mat.Tensor, len(tokens))
	for _, encoder := range m.TokensEncoder {
		for i, encoding := range encoder.EncodeTokens(tokens) {
			encoded[i] = append(encoded[i], encoding)
		}
	}
	return m.Projection.Forward(concat(encoded)...)
}

func concat(xs [][]mat.Tensor) []mat.Tensor {
	fn := func(vectors []mat.Tensor) mat.Tensor {
		if len(vectors) == 1 {
			return vectors[0]
		}
		return ag.Concat(vectors...)
	}

	result := make([]mat.Tensor, len(xs))
	for i, encoding := range xs {
		result[i] = fn(encoding)
	}
	return result
}
