// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &PreNormFeedForwardBlock{}

// PreNormFeedForwardBlock is a feed-forward block with pre-normalization normalization.
type PreNormFeedForwardBlock struct {
	*FeedForwardBlock
}

func init() {
	gob.Register(&PreNormFeedForwardBlock{})
}

// Forward performs the forward pass.
func (m PreNormFeedForwardBlock) Forward(xs []mat.Tensor) []mat.Tensor {
	return ag.Map2(ag.Add, xs, m.FFN.Forward(m.Norm.Forward(xs...)...))
}
