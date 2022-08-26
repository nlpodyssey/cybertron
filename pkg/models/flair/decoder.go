// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/crf"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model = &Decoder{}

type Decoder struct {
	nn.Module
	Scorer *linear.Model
	CRF    *crf.Model
}

func init() {
	gob.Register(&Decoder{})
}

// NewDecoder returns a new model with parameters initialized to zeros.
func NewDecoder(scorer *linear.Model, crf *crf.Model) *Decoder {
	return &Decoder{
		Scorer: scorer,
		CRF:    crf,
	}
}

// Decode performs the viterbi decoding.
func (m *Decoder) Decode(xs []ag.Node) []int {
	return m.CRF.Decode(m.Scorer.Forward(xs...))
}
