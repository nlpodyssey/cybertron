// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bart implements the transformer model introduced by Mike et al., 2019.
// "Bart: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"
// https://arxiv.org/abs/1910.13461
package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/embedding"
)

var _ nn.Model = &Model{}

// Model implements a base Bart encoder-decoder model without any head on top.
type Model struct {
	nn.Module
	// Config is the model configuration.
	Config Config
	// Encoder is the encoder model.
	Encoder *Encoder
	// Decoder is the decoder model.
	Decoder *Decoder
	// Embeddings contains the embeddings shared between the encoder and the decoder.
	Embeddings *embedding.Model
}

func init() {
	gob.Register(&Model{})
}

// New returns a new Bart model.
func New[T float.DType](c Config) *Model {
	emb := embedding.New[T](c.VocabSize, c.DModel)
	return &Model{
		Encoder:    NewEncoder[T](c, embedding.Shared{Model: emb}),
		Decoder:    NewDecoder[T](c, embedding.Shared{Model: emb}),
		Embeddings: emb,
		Config:     c,
	}
}

// Forward performs encoding-decoding over the same input sequence producing the final encoded sequence.
func (m *Model) Forward(inputIDs []int) []ag.Node {
	encoded := m.Encoder.Encode(inputIDs)
	decoded, _ := m.Decoder.Decode(encoded, shiftR(inputIDs, 1), nil, 1)
	return decoded
}

func shiftR[T any](a []T, i int) []T {
	x, b := a[:(len(a)-i)], a[(len(a)-i):]
	return append(b, x...)
}
