package distilbert

import (
	"fmt"
	"github.com/nlpodyssey/cybertron/pkg/models/distilbert"

	"github.com/nlpodyssey/cybertron/pkg/converter/pytorch"
	"github.com/nlpodyssey/spago/mat/float"
)

type paramsPostProcessing[T float.DType] struct {
	*pytorch.ParamsProvider[T]
	c distilbert.Config
}

func fixAttentionLayers[T float.DType](c distilbert.Config) pytorch.PreProcessingFunc[T] {
	return func(params *pytorch.ParamsProvider[T]) error {
		p := paramsPostProcessing[T]{
			ParamsProvider: params,
			c:              c,
		}
		p.fixTransformerSelfAttention()
		return nil
	}
}

func (p *paramsPostProcessing[T]) fixTransformerSelfAttention() {
	for i := 0; i < p.c.NumHiddenLayers; i++ {
		prefix := fmt.Sprintf("distilbert.transformer.layer.%d.attention", i)
		queryWeight := p.Pop(fmt.Sprintf("%s.q_lin.weight", prefix))
		queryBias := p.Pop(fmt.Sprintf("%s.q_lin.bias", prefix))
		keyWeight := p.Pop(fmt.Sprintf("%s.k_lin.weight", prefix))
		keyBias := p.Pop(fmt.Sprintf("%s.k_lin.bias", prefix))
		valueWeight := p.Pop(fmt.Sprintf("%s.v_lin.weight", prefix))
		valueBias := p.Pop(fmt.Sprintf("%s.v_lin.bias", prefix))

		dim := len(queryBias) / p.c.NumAttentionHeads
		dim2 := len(queryBias)
		for j := 0; j < p.c.NumAttentionHeads; j++ {
			from := j * dim
			to := (j + 1) * dim
			newPrefix := fmt.Sprintf("distilbert.transformer.layer.%d.%d.attention", i, j)
			p.Set(fmt.Sprintf("%s.q_lin.weight", newPrefix), queryWeight[from*dim2:to*dim2])
			p.Set(fmt.Sprintf("%s.q_lin.bias", newPrefix), queryBias[from:to])
			p.Set(fmt.Sprintf("%s.k_lin.weight", newPrefix), keyWeight[from*dim2:to*dim2])
			p.Set(fmt.Sprintf("%s.k_lin.bias", newPrefix), keyBias[from:to])
			p.Set(fmt.Sprintf("%s.v_lin.weight", newPrefix), valueWeight[from*dim2:to*dim2])
			p.Set(fmt.Sprintf("%s.v_lin.bias", newPrefix), valueBias[from:to])
		}
	}
}
