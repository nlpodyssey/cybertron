// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

type paramsMap map[string]mat.Tensor

func mapEncoderParams(encoder *bert.Encoder, params paramsMap) {
	for i := 0; i < encoder.Config.NumHiddenLayers; i++ {
		layer := any(encoder.Layers[i]).(*bert.EncoderLayer)
		prefixBase := fmt.Sprintf("bert.encoder.layer.%d", i)

		block1 := layer.SelfAttention
		for j := 0; j < encoder.Config.NumAttentionHeads; j++ {
			attention := block1.Attention.Heads[j]
			prefix := fmt.Sprintf("%s.%d.attention.self", prefixBase, j)
			params[fmt.Sprintf("%s.query.weight", prefix)] = attention.Query.W.Value()
			params[fmt.Sprintf("%s.query.bias", prefix)] = attention.Query.B.Value()
			params[fmt.Sprintf("%s.key.weight", prefix)] = attention.Key.W.Value()
			params[fmt.Sprintf("%s.key.bias", prefix)] = attention.Key.B.Value()
			params[fmt.Sprintf("%s.value.weight", prefix)] = attention.Value.W.Value()
			params[fmt.Sprintf("%s.value.bias", prefix)] = attention.Value.B.Value()
		}
		prefix := fmt.Sprintf("bert.encoder.layer.%d.attention", i)
		params[fmt.Sprintf("%s.output.dense.weight", prefix)] = block1.Attention.OutputMerge.W.Value()
		params[fmt.Sprintf("%s.output.dense.bias", prefix)] = block1.Attention.OutputMerge.B.Value()
		params[fmt.Sprintf("%s.output.LayerNorm.weight", prefix)] = block1.Norm.W.Value()
		params[fmt.Sprintf("%s.output.LayerNorm.bias", prefix)] = block1.Norm.B.Value()

		block2 := layer.FF
		params[fmt.Sprintf("%s.intermediate.dense.weight", prefixBase)] = block2.MLP[0].(*linear.Model).W.Value()
		params[fmt.Sprintf("%s.intermediate.dense.bias", prefixBase)] = block2.MLP[0].(*linear.Model).B.Value()
		params[fmt.Sprintf("%s.output.dense.weight", prefixBase)] = block2.MLP[2].(*linear.Model).W.Value()
		params[fmt.Sprintf("%s.output.dense.bias", prefixBase)] = block2.MLP[2].(*linear.Model).B.Value()
		params[fmt.Sprintf("%s.output.LayerNorm.weight", prefixBase)] = block2.Norm.W.Value()
		params[fmt.Sprintf("%s.output.LayerNorm.bias", prefixBase)] = block2.Norm.B.Value()
	}
}

func mapEmbeddingsLayerNorm(embeddingsNorm *layernorm.Model, params paramsMap) {
	params["bert.embeddings.LayerNorm.weight"] = embeddingsNorm.W.Value()
	params["bert.embeddings.LayerNorm.bias"] = embeddingsNorm.B.Value()
}

func mapPooler(pooler *bert.Pooler, params paramsMap) {
	params["bert.pooler.dense.weight"] = pooler.Model[0].(*linear.Model).W.Value()
	params["bert.pooler.dense.bias"] = pooler.Model[0].(*linear.Model).B.Value()
}

func mapSeqClassifier(model *linear.Model, params paramsMap) {
	params["classifier.weight"] = model.W.Value()
	params["classifier.bias"] = model.B.Value()
}

func mapTokenClassifier(model *linear.Model, params paramsMap) {
	params["classifier.weight"] = model.W.Value()
	params["classifier.bias"] = model.B.Value()
}

// mapProjectionLayer maps the projection layer parameters.
func mapQAClassifier(model *linear.Model, params paramsMap) {
	params["qa_outputs.weight"] = model.W.Value()
	params["qa_outputs.bias"] = model.B.Value()
}

func mapMaskedLM(layers []nn.StandardModel, params paramsMap) {
	params["cls.predictions.transform.dense.weight"] = layers[0].(*linear.Model).W.Value()
	params["cls.predictions.transform.dense.bias"] = layers[0].(*linear.Model).B.Value()
	params["cls.predictions.transform.LayerNorm.weight"] = layers[2].(*layernorm.Model).W.Value()
	params["cls.predictions.transform.LayerNorm.bias"] = layers[2].(*layernorm.Model).B.Value()
	params["cls.predictions.decoder.weight"] = layers[3].(*linear.Model).W.Value()
	params["cls.predictions.decoder.bias"] = layers[3].(*linear.Model).B.Value()
}
