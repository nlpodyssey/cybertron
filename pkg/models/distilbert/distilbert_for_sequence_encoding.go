package distilbert

import (
	"encoding/gob"
	"fmt"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &ModelForSequenceEncoding{}

// ModelForSequenceEncoding implements a DistilBert model for sequence encoding.
type ModelForSequenceEncoding struct {
	nn.Module
	// DistilBert is the fine-tuned DistilBert model.
	DistilBert *Model
}

// PoolingStrategyType defines the strategy to obtain the dense sequence representation
type PoolingStrategyType int

const (
	// ClsToken gets the last encoding state corresponding to [CLS], i.e. the first token (default)
	ClsToken PoolingStrategyType = iota
	// MeanPooling takes the average of the last encoding states
	MeanPooling
	// MaxPooling takes the maximum of the last encoding states
	MaxPooling
	// MeanMaxPooling does MeanPooling and MaxPooling separately and then concat them together
	MeanMaxPooling
)

func init() {
	gob.Register(&ModelForSequenceEncoding{})
}

// NewModelForSequenceEncoding returns a new model for sequence encoding.
func NewModelForSequenceEncoding(distilbert *Model) *ModelForSequenceEncoding {
	return &ModelForSequenceEncoding{
		DistilBert: distilbert,
	}
}

// Encode returns the vector representation for the input sequence.
func (m *ModelForSequenceEncoding) Encode(tokens []string, poolingStrategy PoolingStrategyType) (ag.Node, error) {
	return m.pooling(m.DistilBert.Encode(tokens), poolingStrategy)
}

func (m *ModelForSequenceEncoding) pooling(lastHiddenStates []ag.Node, ps PoolingStrategyType) (ag.Node, error) {
	switch ps {
	case MeanPooling:
		return ag.Mean(lastHiddenStates), nil
	case MaxPooling:
		return ag.Maximum(lastHiddenStates), nil
	case MeanMaxPooling:
		return ag.Concat(ag.Mean(lastHiddenStates), ag.Maximum(lastHiddenStates)), nil
	case ClsToken:
		return lastHiddenStates[0], nil
	default:
		return nil, fmt.Errorf("bert: invalid pooling strategy")
	}
}
