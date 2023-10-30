// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model implements a base Bert encoder model without any head on top.
type Model struct {
	nn.Module
	Embeddings *Embeddings
	Encoder    *Encoder
	Pooler     *Pooler
	Config     Config
}

func init() {
	gob.Register(&Model{})
}

// New returns a new Bert model.
func New[T float.DType](c Config) *Model {
	return &Model{
		Embeddings: NewEmbeddings[T](c),
		Encoder:    NewEncoder[T](c),
		Pooler:     NewPooler[T](c),
		Config:     c,
	}
}

// EncodeTokens produce the encoded representation for the input tokens
func (m *Model) EncodeTokens(tokens []string) []mat.Tensor {
	return m.Encoder.Encode(m.Embeddings.EncodeTokens(tokens))
}
