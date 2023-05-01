// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/embedding"
)

var _ nn.Model = &PositionalEncoder{}

// PositionalEncoderConfig provides configuration settings for a PositionalEncoder model.
type PositionalEncoderConfig struct {
	NumEmbeddings int
	EmbeddingDim  int
	PaddingIDX    int
	Offset        int
	StoreName     string
	Trainable     bool
}

// PositionalEncoder contains positional embeddings fine-tuned during
// the training phase.
type PositionalEncoder struct {
	nn.Module
	// Embeddings contains the embeddings for each position.
	Embeddings *embedding.Model
	// Config contains the configuration settings.
	Config PositionalEncoderConfig
}

func init() {
	gob.Register(&PositionalEncoder{})
}

// NewPositionalEncoder returns a new PositionalEncoder.
func NewPositionalEncoder[T float.DType](config PositionalEncoderConfig) *PositionalEncoder {
	e := embedding.New[T](config.NumEmbeddings, config.EmbeddingDim)

	size := config.EmbeddingDim
	half := (size + (size % 2)) / 2

	for i := 0; i < config.NumEmbeddings+config.Offset; i++ {
		data := make([]T, size)
		for j := 0; j < size; j++ {
			v := T(i) / mat.Pow(10000, 2*T(j/2)/T(size))
			if j%2 == 0 {
				data[j/2] = mat.Sin(v)
			} else {
				data[half+j/2] = mat.Cos(v)
			}
		}
		item, _ := e.Embedding(i)
		item.ReplaceValue(mat.NewVecDense[T](data))
	}
	return &PositionalEncoder{
		Config:     config,
		Embeddings: e,
	}
}

// Encode performs the forward step for each input and returns the result.
func (m *PositionalEncoder) Encode(positions []int) []ag.Node {
	return m.Embeddings.MustEncode(m.shift(positions))
}

// shift returns the shifted positions by the offset.
func (m *PositionalEncoder) shift(positions []int) []int {
	result := make([]int, len(positions))
	for i, pos := range positions {
		result[i] = pos + m.Config.Offset
	}
	return result
}
