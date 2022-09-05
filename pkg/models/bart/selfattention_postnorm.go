// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/attention/multiheadattention"
)

var _ nn.Model = &PostNormSelfAttentionBlock{}

// PostNormSelfAttentionBlock embeds a self-attention block to perform self attention with post normalization.
type PostNormSelfAttentionBlock struct {
	*SelfAttentionBlock
}

func init() {
	gob.Register(&PostNormSelfAttentionBlock{})
}

// Forward performs the forward pass.
func (m PostNormSelfAttentionBlock) Forward(cache multiheadattention.Cache, xs []ag.Node) ([]ag.Node, multiheadattention.Cache) {
	att, _, nextCache := m.Attention.Forward(cache, xs)

	residual := att // reuse the same slice to avoid allocation
	for i := range residual {
		residual[i] = ag.Add(xs[i], att[i])
	}

	norm := m.Norm.Forward(residual...)
	return norm, nextCache
}
