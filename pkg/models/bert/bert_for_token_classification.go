// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model = &ModelForTokenClassification{}

// ModelForTokenClassification implements a Bert model for token classification.
type ModelForTokenClassification struct {
	nn.Module
	// Bart is the fine-tuned BERT model.
	Bert *Model
	// Classifier is the linear layer for sequence classification.
	Classifier *linear.Model
}

func init() {
	gob.Register(&ModelForTokenClassification{})
}

// NewModelForTokenClassification returns a new model for token classification.
func NewModelForTokenClassification[T float.DType](bert *Model) *ModelForTokenClassification {
	return &ModelForTokenClassification{
		Bert:       bert,
		Classifier: linear.New[T](bert.Config.HiddenSize, len(bert.Config.ID2Label)),
	}
}

// Classify returns the logits for each token.
func (m *ModelForTokenClassification) Classify(tokens []string) []mat.Tensor {
	return m.Classifier.Forward(m.Bert.EncodeTokens(tokens)...)
}
