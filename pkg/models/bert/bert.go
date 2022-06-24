// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

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

// Model implements a base Bert encoder model without any head on top.
type Model struct {
	nn.Module
	Embeddings *Embeddings
	Encoder    *Encoder
	Pooler     *Pooler
	Config     Config
}

func init() {
	gob.Register(&Model{})
}

// New returns a new Bert model.
func New[T float.DType](c Config, repo store.Repository) *Model {
	return &Model{
		Embeddings: NewEmbeddings[T](c, repo),
		Encoder:    NewEncoder[T](c),
		Pooler:     NewPooler[T](c),
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
	return err
}

// Encode produce the encoded representation for the input tokens
func (m *Model) Encode(tokens []string) []ag.Node {
	return m.Encoder.Encode(m.Embeddings.Encode(tokens))
}
