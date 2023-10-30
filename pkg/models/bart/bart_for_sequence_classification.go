// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &ModelForSequenceClassification{}

// ModelForSequenceClassification is a model for sequence classification tasks
// which embeds a Bart fine-tuned model.
type ModelForSequenceClassification struct {
	nn.Module
	// Bart is the Bart fine-tuned model.
	Bart *Model
	// Classifier is the final classifier layer.
	Classifier *Classifier
}

func init() {
	gob.Register(&ModelForSequenceClassification{})
}

// NewModelForSequenceClassification returns a new model for sentence-level classification.
func NewModelForSequenceClassification[T float.DType](bart *Model) *ModelForSequenceClassification {
	return &ModelForSequenceClassification{
		Bart: bart,
		Classifier: NewClassifier[T](ClassifierConfig{
			InputSize:     bart.Config.DModel,
			HiddenSize:    bart.Config.DModel,
			OutputSize:    bart.Config.NumLabels,
			PoolerDropout: bart.Config.ClassifierDropout,
		}),
	}
}

// Forward performs the classification using the last transformed state.
func (m *ModelForSequenceClassification) Forward(inputIds []int) mat.Tensor {
	return m.Classifier.Forward(lastState(m.Bart.Forward(inputIds)))
}

// lastState returns the last state of the encoded sequence.
func lastState(xs []mat.Tensor) mat.Tensor {
	return xs[len(xs)-1]
}
