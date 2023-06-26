package distilbert

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &Transformer{}

// Transformer implements a DistilBert encoder.
type Transformer struct {
	nn.Module
	Layers nn.ModuleList[*TransformerLayer]
	Config Config
}

func init() {
	gob.Register(&Transformer{})
}

// NewTransformer returns a new Transformer.
func NewTransformer[T float.DType](c Config) *Transformer {
	layers := make([]*TransformerLayer, c.NumHiddenLayers)
	for i := 0; i < c.NumHiddenLayers; i++ {
		layers[i] = NewEncoderLayer[T](c)
	}
	return &Transformer{
		Layers: layers,
		Config: c,
	}
}

// Encode performs the DistilBert encoding.
func (e *Transformer) Encode(xs []ag.Node) []ag.Node {
	return e.Layers.Forward(xs...)
}
