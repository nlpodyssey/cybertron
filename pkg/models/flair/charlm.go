// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

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

var _ nn.Model = &CharLM{}

const (
	defaultSequenceSeparator = "\n"
	defaultUnknownToken      = "<unk>"
)

// CharLM implements a Character-level Language Model.
type CharLM struct {
	nn.Module
	Config
	Decoder    *linear.Model
	Projection *linear.Model
	RNN        *lstm.Model
	Embeddings *emb.Model[string]
	Vocabulary *vocabulary.Vocabulary
}

func init() {
	gob.Register(&CharLM{})
}

// NewCharLM returns a new character-level language Model, initialized according to
// the given configuration.
func NewCharLM[T float.DType](c Config, repo store.Repository) *CharLM {
	if c.SequenceSeparator == "" {
		c.SequenceSeparator = defaultSequenceSeparator
	}
	if c.UnknownToken == "" {
		c.UnknownToken = defaultUnknownToken
	}
	return &CharLM{
		Config:     c,
		Decoder:    linear.New(c.OutputSize, c.VocabularySize),
		Projection: linear.New(c.HiddenSize, c.OutputSize),
		RNN:        lstm.New(c.EmbeddingSize, c.HiddenSize),
		Embeddings: emb.New[T, string](emb.Config{
			Size:      c.EmbeddingSize,
			StoreName: c.Name,
			Trainable: c.Trainable,
		}, repo),
	}
}

func (m *CharLM) Encode(xs []string) []ag.Node {
	return m.UseProjection(m.RNN.Forward(m.Embeddings.Encode(xs)...))
}

func (m *CharLM) UseProjection(xs []ag.Node) []ag.Node {
	if m.Config.OutputSize > 0 {
		return m.Projection.Forward(xs...)
	}
	return xs
}
