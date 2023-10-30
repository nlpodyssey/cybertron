// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"math"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/embedding"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

type Embeddings struct {
	nn.Module
	// SharedEmbeddings is the shared embedding module.
	SharedEmbeddings embedding.Shared
	// PositionalEncoder is the positional encoder module.
	PositionalEncoder *PositionalEncoder
	// Norm is the normalization module.
	Norm *layernorm.Model
	// ScaleFactor is the scaling factor for the shared embeddings.
	ScaleFactor *nn.Buffer
	// Config is the configuration of the embeddings.
	Config Config
}

// NewEmbeddings returns a new Embeddings.
func NewEmbeddings[T float.DType](c Config, shared embedding.Shared, isDecoder bool) *Embeddings {
	storeName := c.Cybertron.EncoderPositionalEncodingStoreName
	if isDecoder {
		storeName = c.Cybertron.DecoderPositionalEncodingStoreName
	}
	var scaleFactor *nn.Buffer
	if c.ScaleEmbedding {
		scaleFactor = nn.Buf(mat.Scalar(math.Sqrt(float64(c.DModel))))
	}
	return &Embeddings{
		SharedEmbeddings: shared,
		PositionalEncoder: NewPositionalEncoder[T](PositionalEncoderConfig{
			NumEmbeddings: c.MaxPositionEmbeddings,
			EmbeddingDim:  c.DModel,
			PaddingIDX:    c.PadTokenID,
			Offset:        c.Cybertron.PositionalEncoderOffset,
			StoreName:     storeName,
			Trainable:     c.Cybertron.Training,
		}),
		Norm:        layernorm.New[T](c.DModel, 1e-5),
		ScaleFactor: scaleFactor,
		Config:      c,
	}
}

// Encode performs the Bart initial input encoding.
func (m *Embeddings) Encode(inputIDs []int, offset int) []mat.Tensor {
	ys := ag.Map2(ag.Add,
		m.useScaledEmbeddings(m.SharedEmbeddings.MustEncode(inputIDs)),
		m.PositionalEncoder.Encode(makePositions(len(inputIDs), offset)),
	)
	if m.Config.NormalizeEmbedding {
		ys = m.Norm.Forward(ys...)
	}
	return ys
}

// useScaledEmbeddings returns the scaled embeddings.
func (m *Embeddings) useScaledEmbeddings(xs []mat.Tensor) []mat.Tensor {
	if !m.Config.ScaleEmbedding {
		return xs
	}

	ys := make([]mat.Tensor, len(xs))
	for i, x := range xs {
		ys[i] = ag.ProdScalar(x, m.ScaleFactor)
	}
	return ys
}

// makePositions returns a slice of the given size, where each element has
// the same value of its own index position plus the offset.
func makePositions(size, offset int) []int {
	indices := make([]int, size)
	for i := range indices {
		indices[i] = i + offset
	}
	return indices
}
