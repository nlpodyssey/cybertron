// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package charlm

import (
	"encoding/gob"
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/vocabulary"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	emb "github.com/nlpodyssey/spago/nn/embedding"
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
	Embeddings *emb.Model
	Vocabulary *vocabulary.Vocabulary
}

func init() {
	gob.Register(&Model{})
}

// NewCharLM returns a new character-level language Model, initialized according to
// the given configuration.
func NewCharLM[T float.DType](c Config) *Model {
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
		Embeddings: emb.New[T](c.VocabularySize, c.EmbeddingSize),
	}
}

func (m *Model) EncodeTokens(xs []string) []ag.Node {
	indices, err := m.convertStringsToInts(xs)
	if err != nil {
		panic(err) // TODO: return error
	}
	return m.UseProjection(m.RNN.Forward(m.Embeddings.MustEncode(indices)...))
}

func (m *Model) UseProjection(xs []ag.Node) []ag.Node {
	if m.Projection != nil {
		return m.Projection.Forward(xs...)
	}
	return xs
}

func (m *Model) convertStringsToInts(strings []string) ([]int, error) {
	ints := make([]int, len(strings))
	for i, str := range strings {
		value, found := m.Vocabulary.ID(str)
		if !found {
			return nil, fmt.Errorf("word '%s' not found in vocab", str)
		}
		ints[i] = value
	}
	return ints, nil
}
