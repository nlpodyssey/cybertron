// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/attention/multiheadattention"
	"github.com/nlpodyssey/spago/nn/embedding"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

var _ nn.Model = &Decoder{}

// Decoder implements a Bart decoder.
type Decoder struct {
	nn.Module
	// Embeddings is the embedding module.
	Embeddings *Embeddings
	// Layers is the list of decoder layers.
	Layers []*DecoderLayer
	// LayerNorm is the layer normalization module.
	LayerNorm *layernorm.Model
	// Config is the configuration of the decoder.
	Config Config
}

// Cache contains the cache of each DecoderLayer.
// For each layer, the cache contains the queries, keys and values used by the self-attention at index 0 and cross-attention at index 1.
type Cache [][2]multiheadattention.Cache

// Layer returns the cache at the given index.
func (c Cache) Layer(i int) [2]multiheadattention.Cache {
	if len(c) == 0 {
		return [2]multiheadattention.Cache{}
	}
	return c[i]
}

func init() {
	gob.Register(&Decoder{})
}

// NewDecoder returns a new Decoder.
func NewDecoder[T float.DType](c Config, shared embedding.Shared) *Decoder {
	layers := make([]*DecoderLayer, c.DecoderLayers)
	for i := 0; i < c.DecoderLayers; i++ {
		layers[i] = NewDecoderLayer[T](c)
	}
	return &Decoder{
		Embeddings: NewEmbeddings[T](c, shared, true),
		Layers:     layers,
		LayerNorm:  layernorm.New[T](c.DModel, 1e-5),
		Config:     c,
	}
}

// Decode performs the decoding considering the encoder output and the decoder input.
func (m *Decoder) Decode(encoderStates []ag.Node, inputIDs []int, cache Cache, curLen int) ([]ag.Node, Cache) {
	nextCache := make(Cache, len(m.Layers))
	ys := m.Embeddings.Encode(inputIDs, curLen-1)
	for i, layer := range m.Layers {
		ys, nextCache[i] = layer.Forward(cache.Layer(i), ys, encoderStates)
	}
	if m.Config.FinalLayerNorm {
		ys = m.LayerNorm.Forward(ys...)
	}
	return ys, nextCache
}
