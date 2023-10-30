// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

// ResidualNormFeedForward is a feed-forward block with normalization and residual connection.
type ResidualNormFeedForward interface {
	Forward(xs []mat.Tensor) []mat.Tensor
}

var _ nn.Model = &FeedForwardBlock{}

// FeedForwardBlock is a feed-forward block with normalization and residual connection.
type FeedForwardBlock struct {
	nn.Module
	FFN  nn.ModuleList[nn.StandardModel]
	Norm *layernorm.Model
}

func init() {
	gob.Register(&FeedForwardBlock{})
}

// NewFeedForwardBlockConfig is the configuration of a feed-forward block.
type NewFeedForwardBlockConfig struct {
	// Dim is the dimension of the input.
	Dim int
	// HiddenDim is the dimension of the hidden layer.
	HiddenDim int
	// ActivationFunction is the activation function.
	Activation activation.Activation
	// NormalizeBefore is whether to normalize the input before the MLP.
	NormalizeBefore bool
}

// NewFeedForwardBlock returns a new PreNormFeedForwardBlock or PostNormFeedForwardBlock
// depending on the configuration.
func NewFeedForwardBlock[T float.DType](c NewFeedForwardBlockConfig) ResidualNormFeedForward {
	block := &FeedForwardBlock{
		FFN: []nn.StandardModel{
			linear.New[T](c.Dim, c.HiddenDim),
			activation.New(c.Activation),
			linear.New[T](c.HiddenDim, c.Dim),
		},
		Norm: layernorm.New[T](c.Dim, 1e-5),
	}
	if c.NormalizeBefore {
		return PreNormFeedForwardBlock{block}
	}
	return PostNormFeedForwardBlock{block}
}
