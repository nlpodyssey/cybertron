// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
)

var _ nn.Model = &EncoderLayer{}

// EncoderLayer implements a Bart encoder layer.
type EncoderLayer struct {
	nn.Module
	// SelfAttention is the self attention block.
	SelfAttention ResidualNormSelfAttention
	// FF is the feed-forward block with normalization and residual connection.
	FF ResidualNormFeedForward
	// Config is the configuration of the encoder layer.
	Config Config
}

func init() {
	gob.Register(&EncoderLayer{})
}

// NewEncoderLayer returns a new encoder layer.
func NewEncoderLayer[T float.DType](c Config) *EncoderLayer {
	return &EncoderLayer{
		SelfAttention: NewSelfAttentionBlock[T](SelfAttentionBlockConfig{
			Dim:             c.DModel,
			NumOfHeads:      c.EncoderAttentionHeads,
			NormalizeBefore: c.NormalizeBefore,
			UseCausalMask:   false,
		}),
		FF: NewFeedForwardBlock[T](NewFeedForwardBlockConfig{
			Dim:             c.DModel,
			HiddenDim:       c.EncoderFFNDim,
			Activation:      activation.MustActivation(c.ActivationFunction),
			NormalizeBefore: c.NormalizeBefore,
		}),
		Config: c,
	}
}

// Forward performs the forward pass.
func (m *EncoderLayer) Forward(xs ...ag.Node) []ag.Node {
	attention, _ := m.SelfAttention.Forward(nil, xs)
	return m.FF.Forward(attention)
}
