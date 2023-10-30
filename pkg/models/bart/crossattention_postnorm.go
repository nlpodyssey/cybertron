// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/attention/multiheadattention"
)

var _ nn.Model = &PostNormCrossAttentionBlock{}

// PostNormCrossAttentionBlock embeds a cross-attention block to perform cross attention with post normalization
type PostNormCrossAttentionBlock struct {
	*CrossAttentionBlock
}

func init() {
	gob.Register(&PostNormCrossAttentionBlock{})
}

// Forward performs the forward pass.
func (m PostNormCrossAttentionBlock) Forward(cache multiheadattention.Cache, seq1 []mat.Tensor, seq2 []mat.Tensor) ([]mat.Tensor, multiheadattention.Cache) {
	att, _, nextCache := m.Attention.Forward(cache, seq1, seq2)

	residual := att // reuse the same slice to avoid allocation
	for i := range residual {
		residual[i] = ag.Add(seq1[i], att[i])
	}

	norm := m.Norm.Forward(residual...)
	return norm, nextCache
}
