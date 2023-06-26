package distilbert

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/activation"
)

var _ nn.StandardModel = &TransformerLayer{}

// TransformerLayer implements a DistilBert encoder layer.
type TransformerLayer struct {
	nn.Module
	SelfAttention *SelfAttentionBlock
	FF            *FeedForwardBlock
	Config        Config
}

func init() {
	gob.Register(&TransformerLayer{})
}

// NewEncoderLayer returns a new TransformerLayer.
func NewEncoderLayer[T float.DType](c Config) *TransformerLayer {
	return &TransformerLayer{
		SelfAttention: NewSelfAttentionBlock[T](SelfAttentionBlockConfig{
			Dim:        c.HiddenSize,
			NumOfHeads: c.NumAttentionHeads,
		}),
		FF: NewFeedForwardBlock[T](FeedForwardBlockConfig{
			Dim:        c.HiddenSize,
			HiddenDim:  c.IntermediateSize,
			Activation: activation.MustActivation(c.Activation),
		}),
		Config: c,
	}
}

// Forward performs the forward step for each input node and returns the result.
func (m *TransformerLayer) Forward(xs ...ag.Node) []ag.Node {
	return m.FF.Forward(m.SelfAttention.Forward(xs))
}
