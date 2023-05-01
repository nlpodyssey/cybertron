// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn/embedding"
)

type WordEmbeddings struct {
	*embedding.Model
	Vocab map[string]int
}

var _ TokensEncoder = &WordEmbeddings{}

func init() {
	gob.Register(&WordEmbeddings{})
}

func NewWordEmbeddings[T float.DType](vocab map[string]int, embeddingSize int) *WordEmbeddings {
	return &WordEmbeddings{
		Model: embedding.New[T](embeddingSize, len(vocab)),
		Vocab: vocab,
	}
}

// EncodeTokens returns a slice of ag.Node representing the embeddings of the given tokens.
// It first looks up the tokens in the Vocab and then returns the corresponding embeddings.
func (m *WordEmbeddings) EncodeTokens(tokens []string) []ag.Node {
	embeddings := make([]ag.Node, len(tokens))
	for i, token := range tokens {
		idx, ok := m.Vocab[token]
		if !ok {
			panic("flair: unknown token: " + token)
		}
		embeddings[i], _ = m.Model.Embedding(idx) // TODO: check error
	}
	return embeddings
}
