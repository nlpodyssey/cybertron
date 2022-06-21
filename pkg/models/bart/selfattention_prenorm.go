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

var _ nn.Model = &PreNormSelfAttentionBlock{}

// PreNormSelfAttentionBlock embeds a self-attention block to perform self attention with pre-normalization.
type PreNormSelfAttentionBlock struct {
	*SelfAttentionBlock
}

func init() {
	gob.Register(&PreNormSelfAttentionBlock{})
}

// Forward performs the forward pass.
func (m PreNormSelfAttentionBlock) Forward(cache multiheadattention.Cache, xs []ag.Node) ([]ag.Node, multiheadattention.Cache) {
	norm := m.Norm.Forward(xs...)
	att, _, nextCache := m.Attention.Forward(cache, norm)

	residual := att // reuse the same slice to avoid allocation
	for i := range residual {
		residual[i] = ag.Add(xs[i], att[i])
	}

	return residual, nextCache
}
