// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model = &Classifier{}

// ClassifierConfig provides configuration settings for a Bart head for sentence-level
// Classifier model.
type ClassifierConfig struct {
	// InputSize is the input size of the classifier.
	InputSize int
	// HiddenSize is the hidden size of the classifier.
	HiddenSize int
	// OutputSize is the output size of the classifier.
	OutputSize int
	// PoolerDropout is the dropout rate for the classifier.
	PoolerDropout float64
}

// Classifier is a model for Bart head for sentence-level classification tasks.
type Classifier struct {
	nn.Module
	// Config is the configuration of the classifier.
	Config ClassifierConfig
	// Layers is the list of layers of the MLP.
	Layers nn.ModuleList[nn.StandardModel]
}

func init() {
	gob.Register(&Classifier{})
}

// NewClassifier returns a new Classifier.
func NewClassifier[T float.DType](c ClassifierConfig) *Classifier {
	return &Classifier{
		Config: c,
		Layers: []nn.StandardModel{
			// dropout.New(c.PoolerDropout),
			linear.New[T](c.InputSize, c.HiddenSize),
			activation.New(activation.Tanh),
			// dropout.New(c.PoolerDropout),
			linear.New[T](c.HiddenSize, c.OutputSize),
		},
	}
}

// Forward implements the forward pass of the Classifier.
func (m *Classifier) Forward(xs ag.Node) ag.Node {
	return m.Layers.Forward(xs)[0]
}
