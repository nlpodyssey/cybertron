// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
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
func (m *Pooler) Forward(encoded []ag.Node) ag.Node {
	return m.Model.Forward(first(encoded))[0]
}

// first returns only the first node, but waits for the values of all the other nodes.
// This is important to avoid data race on ag.ReleaseGraph(): it is possible that you would
// like to free nodes that depend on other nodes in goroutines still in execution and for
// which no one has ever asked for the Value().
func first(xs []ag.Node) ag.Node {
	for _, x := range xs {
		x.Value()
	}
	return xs[0]
}
