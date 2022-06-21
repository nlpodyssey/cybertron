// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/attention/multiheadattention"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

var _ nn.Model = &SelfAttentionBlock{}

// ResidualNormSelfAttention is a self-attention block with residual normalization.
type ResidualNormSelfAttention interface {
	Forward(cache multiheadattention.Cache, xs []ag.Node) ([]ag.Node, multiheadattention.Cache)
}

// SelfAttentionBlock implements a self-attention block.
type SelfAttentionBlock struct {
	nn.Module
	// Attention is the multi-head attention module.
	Attention *multiheadattention.SelfAttention
	// Norm is the layer normalization module.
	Norm *layernorm.Model
}

func init() {
	gob.Register(&SelfAttentionBlock{})
}

// SelfAttentionBlockConfig is the configuration of a self-attention block.
type SelfAttentionBlockConfig struct {
	// Dim is the dimension of the input and output.
	Dim int
	// NumOfHeads is the number of heads.
	NumOfHeads int
	// NormalizeBefore indicates whether the normalization is applied before or after the attention.
	NormalizeBefore bool
	// UseCausalMask indicates whether to use a causal mask.
	UseCausalMask bool
}

// NewSelfAttentionBlock returns a new PreNormSelfAttentionBlock or PostNormSelfAttentionBlock
// depending on the configuration.
func NewSelfAttentionBlock[T float.DType](c SelfAttentionBlockConfig) ResidualNormSelfAttention {
	block := &SelfAttentionBlock{
		Attention: &multiheadattention.SelfAttention{
			Model: multiheadattention.New[T](c.Dim, c.NumOfHeads, c.UseCausalMask),
		},
		Norm: layernorm.New[T](c.Dim, 1e-5),
	}
	if c.NormalizeBefore {
		return PreNormSelfAttentionBlock{block}
	}
	return PostNormSelfAttentionBlock{block}
}
