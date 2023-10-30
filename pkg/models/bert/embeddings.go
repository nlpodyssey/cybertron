// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"github.com/nlpodyssey/cybertron/pkg/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	emb "github.com/nlpodyssey/spago/nn/embedding"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

// Embeddings implements a Bert input embedding module.
type Embeddings struct {
	nn.Module
	Tokens     *emb.Model // string
	Positions  *emb.Model
	TokenTypes *emb.Model
	Norm       *layernorm.Model
	Projector  *linear.Model
	Config     Config
}

// NewEmbeddings returns a new Bert input embedding module.
func NewEmbeddings[T float.DType](c Config) *Embeddings {
	var projector *linear.Model = nil
	if c.EmbeddingsSize != c.HiddenSize {
		projector = linear.New[T](c.EmbeddingsSize, c.HiddenSize)
	}

	return &Embeddings{
		Tokens:     emb.New[T](c.VocabSize, c.EmbeddingsSize),
		Positions:  emb.New[T](c.MaxPositionEmbeddings, c.EmbeddingsSize),
		TokenTypes: emb.New[T](c.TypeVocabSize, c.EmbeddingsSize),
		Norm:       layernorm.New[T](c.EmbeddingsSize, 1e-5),
		Projector:  projector,
		Config:     c,
	}
}

// EncodeTokens performs the Bert input encoding.
func (m *Embeddings) EncodeTokens(tokens []string) []mat.Tensor {
	var (
		encoded      = m.Tokens.MustEncode([]int{}) // TODO: temporary []int{} should the tokens be []int?
		positions    = m.Positions.MustEncode(indices(len(tokens)))
		tokenType, _ = m.TokenTypes.Embedding(0)
	)

	sequenceIndex := 0
	for i := 0; i < len(tokens); i++ {
		encoded[i] = ag.Sum(encoded[i], positions[i], tokenType)
		if tokens[i] == wordpiecetokenizer.DefaultSequenceSeparator {
			sequenceIndex++
			tokenType, _ = m.TokenTypes.Embedding(sequenceIndex)
		}
	}
	return m.useProjection(m.Norm.Forward(encoded...))
}

// useProjection returns the output of the projector if it is not nil, otherwise the input.
func (m *Embeddings) useProjection(xs []mat.Tensor) []mat.Tensor {
	if m.Projector == nil {
		return xs
	}
	return m.Projector.Forward(xs...)
}

// indices returns a slice of the given size, where each element has
// the same value of its own index position.
func indices(size int) []int {
	idx := make([]int, size)
	for i := range idx {
		idx[i] = i
	}
	return idx
}
