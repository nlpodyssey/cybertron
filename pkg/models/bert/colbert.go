package bert

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/linear"
)

type ColbertModel struct {
	nn.Model
	// Bart is the fine-tuned BERT model.
	Bert *Model
	// Linear is the linear layer for dimensionality reduction
	Linear *linear.Model
}

func init() {
	gob.Register(&ColbertModel{})
}

// NewColbertModel returns a new model for information retrieval using ColBERT
func NewColbertModel[T float.DType](bert *Model) *ColbertModel {
	return &ColbertModel{
		Bert:   bert,
		Linear: linear.New[T](bert.Config.HiddenSize, 128),
		// TODO: read size dimensionality reduction layer from config
		// (artifact-config.metadata , key: dim)
	}
}

// Forward returns the representation for the provided tokens
func (m *ColbertModel) Forward(tokens []string) []ag.Node {
	return m.Linear.Forward(m.Bert.Encode(tokens)...)
}
