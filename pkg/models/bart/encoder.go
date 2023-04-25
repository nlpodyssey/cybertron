// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/embeddings"
	"github.com/nlpodyssey/spago/embeddings/store"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

var _ nn.Model = &Encoder{}

// Encoder implements a Bart encoder.
type Encoder struct {
	nn.Module
	// Embeddings is the embedding layer.
	Embeddings *Embeddings
	// Layers is the list of encoder layers.
	Layers nn.ModuleList[*EncoderLayer]
	// LayerNorm is the layer normalization module.
	LayerNorm *layernorm.Model
	// Config is the configuration of the encoder.
	Config Config
}

func init() {
	gob.Register(&Encoder{})
}

// NewEncoder returns a new Encoder.
func NewEncoder[T float.DType](c Config, repo store.Repository, shared embeddings.Shared[int]) *Encoder {
	layers := make([]*EncoderLayer, c.EncoderLayers)
	for i := 0; i < c.EncoderLayers; i++ {
		layers[i] = NewEncoderLayer[T](c)
	}

	return &Encoder{
		Embeddings: NewEmbeddings[T](c, repo, shared, false),
		Layers:     layers,
		LayerNorm:  layernorm.New[T](c.DModel, 1e-5),
		Config:     c,
	}
}

// Encode performs the Bart encoding.
func (m *Encoder) Encode(inputIDs []int) []ag.Node {
	ys := m.Embeddings.Encode(inputIDs, 0)
	ys = m.Layers.Forward(ys...)
	if m.Config.FinalLayerNorm {
		ys = m.LayerNorm.Forward(ys...)
	}
	return ys // TODO: return all hidden states?
}
