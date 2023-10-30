// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/linear"
)

// Pooler implements a Bert pooler.
type Pooler struct {
	nn.Module
	Model nn.ModuleList[nn.StandardModel]
}

func init() {
	gob.Register(&Pooler{})
}

// NewPooler returns a new Pooler.
func NewPooler[T float.DType](c Config) *Pooler {
	return &Pooler{
		Model: []nn.StandardModel{
			linear.New[T](c.HiddenSize, c.HiddenSize),
			activation.New(activation.Tanh),
		},
	}
}

// Forward applies a linear transformation followed by a Tanh activation to the first `[CLS]` encoded token.
func (m *Pooler) Forward(encoded mat.Tensor) mat.Tensor {
	return m.Model.Forward(encoded)[0]
}
