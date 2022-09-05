// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/attention/multiheadattention"
)

var _ nn.Model = &DecoderLayer{}

// DecoderLayer implements a Bart decoder layer.
type DecoderLayer struct {
	nn.Module
	// SelfAttention is the self-attention block.
	SelfAttention ResidualNormSelfAttention
	// CrossAttention is the cross-attention block.
	CrossAttention ResidualNormCrossAttention
	// FF is the feed-forward block with normalization and residual connection.
	FF ResidualNormFeedForward
	// Config is the configuration of the decoder layer.
	Config Config
}

func init() {
	gob.Register(&DecoderLayer{})
}

// NewDecoderLayer returns a new decoder layer.
func NewDecoderLayer[T float.DType](c Config) *DecoderLayer {
	return &DecoderLayer{
		SelfAttention: NewSelfAttentionBlock[T](SelfAttentionBlockConfig{
			Dim:             c.DModel,
			NumOfHeads:      c.DecoderAttentionHeads,
			NormalizeBefore: c.NormalizeBefore,
			UseCausalMask:   true,
		}),
		CrossAttention: NewCrossAttentionBlock[T](CrossAttentionBlockConfig{
			Dim:             c.DModel,
			NumOfHeads:      c.DecoderAttentionHeads,
			NormalizeBefore: c.NormalizeBefore,
		}),
		FF: NewFeedForwardBlock[T](NewFeedForwardBlockConfig{
			Dim:             c.DModel,
			HiddenDim:       c.DecoderFFNDim,
			Activation:      activation.MustActivation(c.ActivationFunction),
			NormalizeBefore: c.NormalizeBefore,
		}),
		Config: c,
	}
}

// Forward performs the forward pass.
func (m *DecoderLayer) Forward(cache [2]multiheadattention.Cache, seq1 []ag.Node, seq2 []ag.Node) ([]ag.Node, [2]multiheadattention.Cache) {
	var nextCache [2]multiheadattention.Cache
	var selfAttention, crossAttention []ag.Node
	selfAttention, nextCache[0] = m.SelfAttention.Forward(cache[0], seq1)
	crossAttention, nextCache[1] = m.CrossAttention.Forward(cache[1], selfAttention, seq2)
	return m.FF.Forward(crossAttention), nextCache
}
