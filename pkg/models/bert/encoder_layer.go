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
)

var _ nn.StandardModel = &EncoderLayer{}

// EncoderLayer implements a Bert encoder layer.
type EncoderLayer struct {
	nn.Module
	SelfAttention *SelfAttentionBlock
	FF            *FeedForwardBlock
	Config        Config
}

func init() {
	gob.Register(&EncoderLayer{})
}

// NewEncoderLayer returns a new EncoderLayer.
func NewEncoderLayer[T float.DType](c Config) *EncoderLayer {
	return &EncoderLayer{
		SelfAttention: NewSelfAttentionBlock[T](SelfAttentionBlockConfig{
			Dim:        c.HiddenSize,
			NumOfHeads: c.NumAttentionHeads,
		}),
		FF: NewFeedForwardBlock[T](FeedForwardBlockConfig{
			Dim:        c.HiddenSize,
			HiddenDim:  c.IntermediateSize,
			Activation: activation.MustParseActivation(c.HiddenAct),
		}),
		Config: c,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *EncoderLayer) Forward(xs ...mat.Tensor) []mat.Tensor {
	return m.FF.Forward(m.SelfAttention.Forward(xs))
}
