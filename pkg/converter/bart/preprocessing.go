// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/pytorch"
	"github.com/nlpodyssey/cybertron/pkg/models/bart"
	"github.com/nlpodyssey/spago/mat/float"
)

type paramsPostProcessing[T float.DType] struct {
	*pytorch.ParamsProvider[T]
	c bart.Config
}

func fixAttentionLayers[T float.DType](c bart.Config) pytorch.PreProcessingFunc[T] {
	return func(params *pytorch.ParamsProvider[T]) error {
		p := paramsPostProcessing[T]{
			ParamsProvider: params,
			c:              c,
		}
		p.fixEncoderSelfAttention()
		p.fixDecoderSelfAttention()
		p.fixDecoderCrossAttention()
		return nil
	}
}

func (p *paramsPostProcessing[T]) fixEncoderSelfAttention() {
	for i := 0; i < p.c.EncoderLayers; i++ {
		prefix := fmt.Sprintf("model.encoder.layers.%d.self_attn", i)
		queryWeight := p.Pop(fmt.Sprintf("%s.q_proj.weight", prefix))
		queryBias := p.Pop(fmt.Sprintf("%s.q_proj.bias", prefix))
		keyWeight := p.Pop(fmt.Sprintf("%s.k_proj.weight", prefix))
		keyBias := p.Pop(fmt.Sprintf("%s.k_proj.bias", prefix))
		valueWeight := p.Pop(fmt.Sprintf("%s.v_proj.weight", prefix))
		valueBias := p.Pop(fmt.Sprintf("%s.v_proj.bias", prefix))
		dim := len(queryBias) / p.c.EncoderAttentionHeads
		dim2 := len(queryBias)
		for j := 0; j < p.c.EncoderAttentionHeads; j++ {
			from := j * dim
			to := (j + 1) * dim
			newPrefix := fmt.Sprintf("model.encoder.layers.%d.%d.self_attn", i, j)
			p.Set(fmt.Sprintf("%s.q_proj.weight", newPrefix), queryWeight[from*dim2:to*dim2])
			p.Set(fmt.Sprintf("%s.q_proj.bias", newPrefix), queryBias[from:to])
			p.Set(fmt.Sprintf("%s.k_proj.weight", newPrefix), keyWeight[from*dim2:to*dim2])
			p.Set(fmt.Sprintf("%s.k_proj.bias", newPrefix), keyBias[from:to])
			p.Set(fmt.Sprintf("%s.v_proj.weight", newPrefix), valueWeight[from*dim2:to*dim2])
			p.Set(fmt.Sprintf("%s.v_proj.bias", newPrefix), valueBias[from:to])
		}
	}
}

func (p *paramsPostProcessing[T]) fixDecoderSelfAttention() {
	for i := 0; i < p.c.DecoderLayers; i++ {
		prefix := fmt.Sprintf("model.decoder.layers.%d.self_attn", i)
		queryWeight := p.Pop(fmt.Sprintf("%s.q_proj.weight", prefix))
		queryBias := p.Pop(fmt.Sprintf("%s.q_proj.bias", prefix))
		keyWeight := p.Pop(fmt.Sprintf("%s.k_proj.weight", prefix))
		keyBias := p.Pop(fmt.Sprintf("%s.k_proj.bias", prefix))
		valueWeight := p.Pop(fmt.Sprintf("%s.v_proj.weight", prefix))
		valueBias := p.Pop(fmt.Sprintf("%s.v_proj.bias", prefix))
		dim := len(queryBias) / p.c.DecoderAttentionHeads
		dim2 := len(queryBias)
		for j := 0; j < p.c.DecoderAttentionHeads; j++ {
			from := j * dim
			to := (j + 1) * dim
			newPrefix := fmt.Sprintf("model.decoder.layers.%d.%d.self_attn", i, j)
			p.Set(fmt.Sprintf("%s.q_proj.weight", newPrefix), queryWeight[from*dim2:to*dim2])
			p.Set(fmt.Sprintf("%s.q_proj.bias", newPrefix), queryBias[from:to])
			p.Set(fmt.Sprintf("%s.k_proj.weight", newPrefix), keyWeight[from*dim2:to*dim2])
			p.Set(fmt.Sprintf("%s.k_proj.bias", newPrefix), keyBias[from:to])
			p.Set(fmt.Sprintf("%s.v_proj.weight", newPrefix), valueWeight[from*dim2:to*dim2])
			p.Set(fmt.Sprintf("%s.v_proj.bias", newPrefix), valueBias[from:to])
		}
	}
}

func (p *paramsPostProcessing[T]) fixDecoderCrossAttention() {
	for i := 0; i < p.c.DecoderLayers; i++ {
		prefix := fmt.Sprintf("model.decoder.layers.%d.encoder_attn", i)
		queryWeight := p.Pop(fmt.Sprintf("%s.q_proj.weight", prefix))
		queryBias := p.Pop(fmt.Sprintf("%s.q_proj.bias", prefix))
		keyWeight := p.Pop(fmt.Sprintf("%s.k_proj.weight", prefix))
		keyBias := p.Pop(fmt.Sprintf("%s.k_proj.bias", prefix))
		valueWeight := p.Pop(fmt.Sprintf("%s.v_proj.weight", prefix))
		valueBias := p.Pop(fmt.Sprintf("%s.v_proj.bias", prefix))
		dim := len(queryBias) / p.c.DecoderAttentionHeads
		dim2 := len(queryBias)
		for j := 0; j < p.c.DecoderAttentionHeads; j++ {
			from := j * dim
			to := (j + 1) * dim
			newPrefix := fmt.Sprintf("model.decoder.layers.%d.%d.encoder_attn", i, j)
			p.Set(fmt.Sprintf("%s.q_proj.weight", newPrefix), queryWeight[from*dim2:to*dim2])
			p.Set(fmt.Sprintf("%s.q_proj.bias", newPrefix), queryBias[from:to])
			p.Set(fmt.Sprintf("%s.k_proj.weight", newPrefix), keyWeight[from*dim2:to*dim2])
			p.Set(fmt.Sprintf("%s.k_proj.bias", newPrefix), keyBias[from:to])
			p.Set(fmt.Sprintf("%s.v_proj.weight", newPrefix), valueWeight[from*dim2:to*dim2])
			p.Set(fmt.Sprintf("%s.v_proj.bias", newPrefix), valueBias[from:to])
		}
	}
}
