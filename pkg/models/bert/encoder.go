// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Encoder{}

// Encoder implements a Bert encoder.
type Encoder struct {
	nn.Module
	Layers nn.ModuleList[*EncoderLayer]
	Config Config
}

func init() {
	gob.Register(&Encoder{})
}

// NewEncoder returns a new Encoder.
func NewEncoder[T float.DType](c Config) *Encoder {
	layers := make([]*EncoderLayer, c.NumHiddenLayers)
	for i := 0; i < c.NumHiddenLayers; i++ {
		layers[i] = NewEncoderLayer[T](c)
	}
	return &Encoder{
		Layers: layers,
		Config: c,
	}
}

// Encode performs the Bert encoding.
func (e *Encoder) Encode(xs []ag.Node) []ag.Node {
	return e.Layers.Forward(xs...)
}
