// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/mat"
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
func (m *Decoder) Decode(xs []mat.Tensor) ([]int, []float64) {
	scores := m.Scorer.Forward(xs...)
	return m.CRF.Decode(scores), bestScores(scores)
}

func bestScores(scores []mat.Tensor) []float64 {
	bests := make([]float64, len(scores))
	for i, item := range scores {
		bests[i] = item.Value().(mat.Matrix).Softmax().Max().Item().F64()
	}
	return bests
}
