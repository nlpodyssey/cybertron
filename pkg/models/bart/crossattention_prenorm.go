// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/attention/multiheadattention"
)

var _ nn.Model = &PreNormCrossAttentionBlock{}

// PreNormCrossAttentionBlock embeds a cross-attention block to perform cross attention with pre normalization
type PreNormCrossAttentionBlock struct {
	*CrossAttentionBlock
}

func init() {
	gob.Register(&PreNormCrossAttentionBlock{})
}

func (m PreNormCrossAttentionBlock) Forward(cache multiheadattention.Cache, seq1 []ag.Node, seq2 []ag.Node) ([]ag.Node, multiheadattention.Cache) {
	norm := m.Norm.Forward(seq1...)
	att, _, nextCache := m.Attention.Forward(cache, norm, seq2)

	residual := att // reuse the same slice to avoid allocation
	for i := range residual {
		residual[i] = ag.Add(seq1[i], att[i])
	}

	return residual, nextCache
}
