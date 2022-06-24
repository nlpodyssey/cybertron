// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/pytorch"
	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/spago/mat/float"
)

type paramsPostProcessing[T float.DType] struct {
	*pytorch.ParamsProvider[T]
	c bert.Config
}

func fixAttentionLayers[T float.DType](c bert.Config) pytorch.PreProcessingFunc[T] {
	return func(params *pytorch.ParamsProvider[T]) error {
		p := paramsPostProcessing[T]{
			ParamsProvider: params,
			c:              c,
		}
		p.fixEncoderSelfAttention()
		return nil
	}
}

func (p *paramsPostProcessing[T]) fixEncoderSelfAttention() {
	for i := 0; i < p.c.NumHiddenLayers; i++ {
		prefix := fmt.Sprintf("bert.encoder.layer.%d.attention.self", i)
		queryWeight := p.Pop(fmt.Sprintf("%s.query.weight", prefix))
		queryBias := p.Pop(fmt.Sprintf("%s.query.bias", prefix))
		keyWeight := p.Pop(fmt.Sprintf("%s.key.weight", prefix))
		keyBias := p.Pop(fmt.Sprintf("%s.key.bias", prefix))
		valueWeight := p.Pop(fmt.Sprintf("%s.value.weight", prefix))
		valueBias := p.Pop(fmt.Sprintf("%s.value.bias", prefix))

		dim := len(queryBias) / p.c.NumAttentionHeads
		dim2 := len(queryBias)
		for j := 0; j < p.c.NumAttentionHeads; j++ {
			from := j * dim
			to := (j + 1) * dim
			newPrefix := fmt.Sprintf("bert.encoder.layer.%d.%d.attention.self", i, j)
			p.Set(fmt.Sprintf("%s.query.weight", newPrefix), queryWeight[from*dim2:to*dim2])
			p.Set(fmt.Sprintf("%s.query.bias", newPrefix), queryBias[from:to])
			p.Set(fmt.Sprintf("%s.key.weight", newPrefix), keyWeight[from*dim2:to*dim2])
			p.Set(fmt.Sprintf("%s.key.bias", newPrefix), keyBias[from:to])
			p.Set(fmt.Sprintf("%s.value.weight", newPrefix), valueWeight[from*dim2:to*dim2])
			p.Set(fmt.Sprintf("%s.value.bias", newPrefix), valueBias[from:to])
		}
	}
}
