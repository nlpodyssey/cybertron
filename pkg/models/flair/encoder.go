// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/birnn"
)

var _ nn.Model = &Encoder{}

type Encoder struct {
	nn.Module
	Embeddings *Embeddings
	BiRNN      *birnn.Model
}

func init() {
	gob.Register(&Encoder{})
}

// NewEncoder returns a new model with parameters initialized to zeros.
func NewEncoder(embeddings *Embeddings, biRNN *birnn.Model) *Encoder {
	return &Encoder{
		Embeddings: embeddings,
		BiRNN:      biRNN,
	}
}

// Encode encodes the sequence of tokens.
func (m *Encoder) Encode(tokens []string) []ag.Node {
	return m.BiRNN.Forward(m.Embeddings.Encode(tokens)...)
}
