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
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

var _ nn.Model = &FeedForwardBlock{}

// FeedForwardBlock implements a Bert feed forward block.
type FeedForwardBlock struct {
	nn.Module
	MLP  nn.ModuleList[nn.StandardModel]
	Norm *layernorm.Model
}

func init() {
	gob.Register(&FeedForwardBlock{})
}

// FeedForwardBlockConfig is the configuration for the FeedForwardBlock.
type FeedForwardBlockConfig struct {
	Dim        int
	HiddenDim  int
	Activation activation.Name
}

// NewFeedForwardBlock returns a new FeedForwardBlock.
func NewFeedForwardBlock[T float.DType](c FeedForwardBlockConfig) *FeedForwardBlock {
	return &FeedForwardBlock{
		MLP: []nn.StandardModel{
			linear.New[T](c.Dim, c.HiddenDim),
			activation.New(c.Activation),
			linear.New[T](c.HiddenDim, c.Dim),
		},
		Norm: layernorm.New[T](c.Dim, 1e-5),
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m FeedForwardBlock) Forward(xs []ag.Node) []ag.Node {
	return m.Norm.Forward(ag.Map2Concurrent(ag.Add, xs, m.MLP.Forward(xs...))...)
}
