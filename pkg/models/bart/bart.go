// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package bart implements the transformer model introduced by Mike et al., 2019.
// "Bart: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension"
// https://arxiv.org/abs/1910.13461
package bart

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/embeddings"
	"github.com/nlpodyssey/spago/embeddings/store"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Model{}

// Model implements a base Bart encoder-decoder model without any head on top.
type Model struct {
	nn.Module
	// Config is the model configuration.
	Config Config
	// Encoder is the encoder model.
	Encoder *Encoder
	// Decoder is the decoder model.
	Decoder *Decoder
	// Embeddings contains the embeddings shared between the encoder and the decoder.
	Embeddings *embeddings.Model[int]
}

func init() {
	gob.Register(&Model{})
}

// New returns a new Bart model.
func New[T float.DType](c Config, repo store.Repository) *Model {
	emb := embeddings.New[T, int](
		embeddings.Config{
			Size:      c.DModel,
			StoreName: c.Cybertron.SharedEmbeddingsStoreName,
			Trainable: c.Cybertron.Training,
		}, repo,
	)
	return &Model{
		Encoder:    NewEncoder[T](c, repo, embeddings.Shared[int]{emb}),
		Decoder:    NewDecoder[T](c, repo, embeddings.Shared[int]{emb}),
		Embeddings: emb,
		Config:     c,
	}
}

// SetEmbeddings sets the embeddings of the model.
func (m *Model) SetEmbeddings(repo *diskstore.Repository) (err error) {
	nn.Apply(m, func(model nn.Model, name string) {
		switch em := model.(type) {
		case *embeddings.Model[[]byte], *embeddings.Model[int], *embeddings.Model[string]:
			// In order to avoid setting the repository to shared embeddings,
			// it is essential to perform the check on the concrete types.
			// However, use duck-typing to avoid having to do a separate case per key type.
			if e := em.(interface {
				UseRepository(repo store.Repository) error
			}).UseRepository(repo); e != nil && err == nil {
				err = e
			}
		}
	})
	if err != nil {
		return err
	}
	m.Encoder.Embeddings.SharedEmbeddings = embeddings.Shared[int]{m.Embeddings}
	m.Decoder.Embeddings.SharedEmbeddings = embeddings.Shared[int]{m.Embeddings}
	return nil
}

// Forward performs encoding-decoding over the same input sequence producing the final encoded sequence.
func (m *Model) Forward(inputIDs []int) []ag.Node {
	encoded := m.Encoder.Encode(inputIDs)
	decoded, _ := m.Decoder.Decode(encoded, shiftR(inputIDs, 1), nil, 1)
	return decoded
}

func shiftR[T any](a []T, i int) []T {
	x, b := a[:(len(a)-i)], a[(len(a)-i):]
	return append(b, x...)
}
