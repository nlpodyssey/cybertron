package distilbert

import (
	"encoding/gob"

	"github.com/nlpodyssey/cybertron/pkg/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

var _ nn.Model = &ModelForMaskedLM{}

// ModelForMaskedLM implements a DistilBert model for masked language modeling.
type ModelForMaskedLM struct {
	nn.Module
	// DistilBert is the fine-tuned model.
	DistilBert *Model
	// Layers contains the feedforward layers for masked language modeling.
	Layers nn.ModuleList[nn.StandardModel]
}

func init() {
	gob.Register(&ModelForMaskedLM{})
}

// NewModelForMaskedLM returns a new model for masked language model.
func NewModelForMaskedLM[T float.DType](distilbert *Model) *ModelForMaskedLM {
	c := distilbert.Config
	return &ModelForMaskedLM{
		DistilBert: distilbert,
		Layers: []nn.StandardModel{
			linear.New[T](c.HiddenSize, c.HiddenSize),
			activation.New(activation.MustActivation(c.Activation)),
			layernorm.New[T](c.HiddenSize, 1e-5),
			linear.New[T](c.HiddenSize, c.VocabSize),
		},
	}
}

// Predict returns the predictions for the token associated to the masked nodes.
func (m *ModelForMaskedLM) Predict(tokens []string) map[int]ag.Node {
	encoded := evaluate(m.DistilBert.Encode(tokens)...)
	result := make(map[int]ag.Node)
	for _, id := range masked(tokens) {
		result[id] = m.Layers.Forward(encoded[id])[0]
	}
	return result
}

func evaluate(xs ...ag.Node) []ag.Node {
	for _, x := range xs {
		x.Value()
	}
	return xs
}

func masked(tokens []string) []int {
	result := make([]int, 0)
	for i := range tokens {
		if tokens[i] == wordpiecetokenizer.DefaultMaskToken {
			result = append(result, i) // target tokens
		}
	}
	return result
}
