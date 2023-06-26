package distilbert

import (
	"fmt"
	"github.com/nlpodyssey/cybertron/pkg/models/distilbert"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

type paramsMap map[string]mat.Matrix

func mapTransformerParams(transformer *distilbert.Transformer, params paramsMap) {
	for i := 0; i < transformer.Config.NumHiddenLayers; i++ {
		layer := any(transformer.Layers[i]).(*distilbert.TransformerLayer)
		prefixBase := fmt.Sprintf("distilbert.transformer.layer.%d", i)

		block1 := layer.SelfAttention
		for j := 0; j < transformer.Config.NumAttentionHeads; j++ {
			attention := block1.Attention.Heads[j]
			prefix := fmt.Sprintf("%s.%d.attention", prefixBase, j)
			params[fmt.Sprintf("%s.q_lin.weight", prefix)] = attention.Query.W.Value()
			params[fmt.Sprintf("%s.q_lin.bias", prefix)] = attention.Query.B.Value()
			params[fmt.Sprintf("%s.k_lin.weight", prefix)] = attention.Key.W.Value()
			params[fmt.Sprintf("%s.k_lin.bias", prefix)] = attention.Key.B.Value()
			params[fmt.Sprintf("%s.v_lin.weight", prefix)] = attention.Value.W.Value()
			params[fmt.Sprintf("%s.v_lin.bias", prefix)] = attention.Value.B.Value()
		}
		prefix := fmt.Sprintf("distilbert.transformer.layer.%d.attention", i)
		params[fmt.Sprintf("%s.out_lin.weight", prefix)] = block1.Attention.OutputMerge.W.Value()
		params[fmt.Sprintf("%s.out_lin.bias", prefix)] = block1.Attention.OutputMerge.B.Value()

		params[fmt.Sprintf("%s.sa_layer_norm.weight", prefixBase)] = block1.Norm.W.Value()
		params[fmt.Sprintf("%s.sa_layer_norm.bias", prefixBase)] = block1.Norm.B.Value()

		block2 := layer.FF
		params[fmt.Sprintf("%s.ffn.lin1.weight", prefixBase)] = block2.ModuleList[0].(*linear.Model).W.Value()
		params[fmt.Sprintf("%s.ffn.lin1.bias", prefixBase)] = block2.ModuleList[0].(*linear.Model).B.Value()
		params[fmt.Sprintf("%s.ffn.lin2.weight", prefixBase)] = block2.ModuleList[2].(*linear.Model).W.Value()
		params[fmt.Sprintf("%s.ffn.lin2.bias", prefixBase)] = block2.ModuleList[2].(*linear.Model).B.Value()
		params[fmt.Sprintf("%s.output_layer_norm.weight", prefixBase)] = block2.Norm.W.Value()
		params[fmt.Sprintf("%s.output_layer_norm.bias", prefixBase)] = block2.Norm.B.Value()
	}
}

func mapEmbeddingsLayerNorm(embeddingsNorm *layernorm.Model, params paramsMap) {
	params["distilbert.embeddings.LayerNorm.weight"] = embeddingsNorm.W.Value()
	params["distilbert.embeddings.LayerNorm.bias"] = embeddingsNorm.B.Value()
}

func mapMaskedLM(layers []nn.StandardModel, params paramsMap) {
	params["vocab_transform.weight"] = layers[0].(*linear.Model).W.Value()
	params["vocab_transform.bias"] = layers[0].(*linear.Model).B.Value()
	params["vocab_projector.weight"] = layers[3].(*linear.Model).W.Value()
	params["vocab_projector.bias"] = layers[3].(*linear.Model).B.Value()
	params["vocab_layer_norm.weight"] = layers[2].(*layernorm.Model).W.Value()
	params["vocab_layer_norm.bias"] = layers[2].(*layernorm.Model).B.Value()
}
