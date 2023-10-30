// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/gob"
	"fmt"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &ModelForQuestionAnswering{}

// ModelForSequenceEncoding implements a Bert model for sequence encoding.
type ModelForSequenceEncoding struct {
	nn.Module
	// Bart is the fine-tuned BERT model.
	Bert *Model
}

// PoolingStrategyType defines the strategy to obtain the dense sequence representation
type PoolingStrategyType int

const (
	// ClsTokenPooling gets the last encoding state corresponding to [CLS], i.e. the first token (default)
	ClsTokenPooling PoolingStrategyType = iota
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
func NewModelForSequenceEncoding(bert *Model) *ModelForSequenceEncoding {
	return &ModelForSequenceEncoding{
		Bert: bert,
	}
}

// Encode returns the vector representation for the input sequence.
func (m *ModelForSequenceEncoding) Encode(tokens []string, poolingStrategy PoolingStrategyType) (mat.Tensor, error) {
	return m.pooling(m.Bert.EncodeTokens(tokens), poolingStrategy)
}

func (m *ModelForSequenceEncoding) pooling(lastHiddenStates []mat.Tensor, ps PoolingStrategyType) (mat.Tensor, error) {
	switch ps {
	case MeanPooling:
		return ag.Mean(lastHiddenStates), nil
	case MaxPooling:
		return ag.Maximum(lastHiddenStates), nil
	case MeanMaxPooling:
		return ag.Concat(ag.Mean(lastHiddenStates), ag.Maximum(lastHiddenStates)), nil
	case ClsTokenPooling:
		return m.Bert.Pooler.Forward(lastHiddenStates[0]), nil
	default:
		return nil, fmt.Errorf("bert: invalid pooling strategy")
	}
}
