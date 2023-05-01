// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model = &ModelForQuestionAnswering{}

// ModelForQuestionAnswering implements span classification for extractive question-answering tasks.
// It uses a linear layers to compute "span start logits" and "span end logits".
type ModelForQuestionAnswering struct {
	nn.Module
	// Bart is the fine-tuned BERT model.
	Bert *Model
	// Classifier is the linear layer for span classification.
	Classifier *linear.Model
}

func init() {
	gob.Register(&ModelForQuestionAnswering{})
}

// NewModelForQuestionAnswering returns a new model for question-answering.
func NewModelForQuestionAnswering[T float.DType](bert *Model) *ModelForQuestionAnswering {
	return &ModelForQuestionAnswering{
		Bert:       bert,
		Classifier: linear.New[T](bert.Config.HiddenSize, 2),
	}
}

// Answer returns the "span start logits" and "span end logits".
func (m *ModelForQuestionAnswering) Answer(tokens []string) (starts, ends []ag.Node) {
	for _, y := range m.Classifier.Forward(m.Bert.EncodeTokens(tokens)...) {
		starts = append(starts, ag.AtVec(y, 0))
		ends = append(ends, ag.AtVec(y, 1))
	}
	return
}
