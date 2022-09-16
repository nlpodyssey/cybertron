// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package flair provides an implementation of a Bidirectional Recurrent Neural Network (BiRNN)
// with a Conditional Random Fields (CRF) on top.
package flair

import (
	"github.com/nlpodyssey/spago/embeddings/store"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/spago/nn"
)

// Model is an encoder-decoder architecture for sequence labeling.
type Model struct {
	nn.Module
	Encoder *Encoder
	Decoder *Decoder
}

// SetEmbeddings sets the embeddings of the model.
func (m *Model) SetEmbeddings(repo *diskstore.Repository) (err error) {
	nn.Apply(m, func(model nn.Model, name string) {
		if emb, ok := model.(interface {
			UseRepository(repo store.Repository) error
		}); ok {
			if e := emb.UseRepository(repo); e != nil && err == nil {
				err = e
			}
		}
	})
	if err != nil {
		return err
	}
	return nil
}

// Forward annotates the input tokens.
func (m *Model) Forward(tokens []string) ([]int, []float64) {
	return m.Decoder.Decode(m.Encoder.Encode(tokens))
}
