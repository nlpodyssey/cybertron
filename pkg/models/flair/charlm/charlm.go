// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package charlm

import (
	"encoding/gob"

	"github.com/nlpodyssey/cybertron/pkg/vocabulary"
	"github.com/nlpodyssey/spago/ag"
	emb "github.com/nlpodyssey/spago/embeddings"
	"github.com/nlpodyssey/spago/embeddings/store"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/nlpodyssey/spago/nn/recurrent/lstm"
)

var _ nn.Model = &Model{}

const (
	defaultSequenceSeparator = "\n"
	defaultUnknownToken      = "<unk>"
)

// Model implements a Character-level Language Model.
type Model struct {
	nn.Module
	Config
	Decoder    *linear.Model
	Projection *linear.Model
	RNN        *lstm.Model
	Embeddings *emb.Model[string]
	Vocabulary *vocabulary.Vocabulary
}

func init() {
	gob.Register(&Model{})
}

// NewCharLM returns a new character-level language Model, initialized according to
// the given configuration.
func NewCharLM[T float.DType](c Config, repo store.Repository) *Model {
	if c.SequenceSeparator == "" {
		c.SequenceSeparator = defaultSequenceSeparator
	}
	if c.UnknownToken == "" {
		c.UnknownToken = defaultUnknownToken
	}
	return &Model{
		Config:     c,
		Decoder:    linear.New[T](c.OutputSize, c.VocabularySize),
		Projection: linear.New[T](c.HiddenSize, c.OutputSize),
		RNN:        lstm.New[T](c.EmbeddingSize, c.HiddenSize),
		Embeddings: emb.New[T, string](emb.Config{
			Size:      c.EmbeddingSize,
			StoreName: c.Name,
			Trainable: c.Trainable,
		}, repo),
	}
}

func (m *Model) Encode(xs []string) []ag.Node {
	return m.UseProjection(m.RNN.Forward(m.Embeddings.Encode(xs)...))
}

func (m *Model) UseProjection(xs []ag.Node) []ag.Node {
	if m.Projection != nil {
		return m.Projection.Forward(xs...)
	}
	return xs
}
