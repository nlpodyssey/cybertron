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

var _ nn.Model = &PostNormFeedForwardBlock{}

// PostNormFeedForwardBlock is a feed-forward block with post-normalization normalization.
type PostNormFeedForwardBlock struct {
	*FeedForwardBlock
}

func init() {
	gob.Register(&PostNormFeedForwardBlock{})
}

// Forward performs the forward pass.
func (m PostNormFeedForwardBlock) Forward(xs []mat.Tensor) []mat.Tensor {
	return m.Norm.Forward(ag.Map2(ag.Add, xs, m.FFN.Forward(xs...))...)
}
