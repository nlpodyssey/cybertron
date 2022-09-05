// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
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
func (m PostNormFeedForwardBlock) Forward(xs []ag.Node) []ag.Node {
	return m.Norm.Forward(ag.Map2(ag.Add, xs, nn.Forward(m.FFN)(xs...))...)
}
