package distilbert

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

// Model implements a base DistilBert encoder model without any head on top.
type Model struct {
	nn.Module
	Embeddings  *Embeddings
	Transformer *Transformer
	Config      Config
}

func init() {
	gob.Register(&Model{})
}

// New returns a new DistilBert model.
func New[T float.DType](c Config, repo store.Repository) *Model {
	return &Model{
		Embeddings:  NewEmbeddings[T](c, repo),
		Transformer: NewTransformer[T](c),
		Config:      c,
	}
}

// SetEmbeddings sets the embeddings of the model.
func (m *Model) SetEmbeddings(repo *diskstore.Repository) (err error) {
	nn.Apply(m, func(model nn.Model) {
		switch em := model.(type) {
		case *embeddings.Model[[]byte], *embeddings.Model[int], *embeddings.Model[string]:
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
	return m.Transformer.Encode(m.Embeddings.Encode(tokens))
}
