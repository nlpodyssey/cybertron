// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"

	"github.com/nlpodyssey/cybertron/pkg/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

var _ nn.Model = &ModelForTokenClassification{}

// ModelForMaskedLM implements a Bert model for masked language modeling.
type ModelForMaskedLM struct {
	nn.Module
	// Bart is the fine-tuned BERT model.
	Bert *Model
	// Layers contains the feedforward layers for masked language modeling.
	Layers nn.ModuleList[nn.StandardModel]
}

func init() {
	gob.Register(&ModelForMaskedLM{})
}

// NewModelForMaskedLM returns a new model for masked language model.
func NewModelForMaskedLM[T float.DType](bert *Model) *ModelForMaskedLM {
	c := bert.Config
	return &ModelForMaskedLM{
		Bert: bert,
		Layers: []nn.StandardModel{
			linear.New[T](c.HiddenSize, c.HiddenSize),
			activation.New(activation.MustParseActivation(c.HiddenAct)),
			layernorm.New[T](c.HiddenSize, 1e-5),
			linear.New[T](c.HiddenSize, c.VocabSize),
		},
	}
}

// Predict returns the predictions for the token associated to the masked nodes.
func (m *ModelForMaskedLM) Predict(tokens []string) map[int]mat.Tensor {
	encoded := evaluate(m.Bert.EncodeTokens(tokens)...)
	result := make(map[int]mat.Tensor)
	for _, id := range masked(tokens) {
		result[id] = m.Layers.Forward(encoded[id])[0]
	}
	return result
}

func evaluate(xs ...mat.Tensor) []mat.Tensor {
	for _, x := range xs {
		x.Value()
	}
	return xs
}

func masked(tokens []string) []int {
	result := make([]int, 0)
	for i := range tokens {
		if tokens[i] == wordpiecetokenizer.DefaultMaskToken {
			result = append(result, i) // target tokens
		}
	}
	return result
}
