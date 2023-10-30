// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/models/bart"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn/linear"
)

// paramsMap is a map of parameters.
type paramsMap map[string]mat.Tensor

// mapClassifier maps the classifier parameters.
func mapClassifier(classifier *bart.Classifier, params paramsMap) {
	params["classification_head.dense.weight"] = classifier.Layers[0].(*linear.Model).W.Value()
	params["classification_head.dense.bias"] = classifier.Layers[0].(*linear.Model).B.Value()
	params["classification_head.out_proj.weight"] = classifier.Layers[2].(*linear.Model).W.Value()
	params["classification_head.out_proj.bias"] = classifier.Layers[2].(*linear.Model).B.Value()
}

// mapProjectionLayer maps the projection layer parameters.
func mapProjectionLayer(model *linear.Model, params paramsMap) {
	params["model.shared.weight"] = model.W.Value()
	params["final_logits_bias"] = model.B.Value()
}

// mapEncoderParams maps the encoder parameters.
func mapEncoderParams(encoder *bart.Encoder, params paramsMap) {
	for i := 0; i < encoder.Config.EncoderLayers; i++ {
		layer := any(encoder.Layers[i]).(*bart.EncoderLayer)
		prefixBase := fmt.Sprintf("model.encoder.layers.%d", i)

		block1 := resolveSelfAttentionBlock(layer.SelfAttention)
		for j := 0; j < encoder.Config.EncoderAttentionHeads; j++ {
			attention := block1.Attention.Heads[j]
			prefix := fmt.Sprintf("%s.%d.self_attn", prefixBase, j)
			params[fmt.Sprintf("%s.q_proj.weight", prefix)] = attention.Query.W.Value()
			params[fmt.Sprintf("%s.q_proj.bias", prefix)] = attention.Query.B.Value()
			params[fmt.Sprintf("%s.k_proj.weight", prefix)] = attention.Key.W.Value()
			params[fmt.Sprintf("%s.k_proj.bias", prefix)] = attention.Key.B.Value()
			params[fmt.Sprintf("%s.v_proj.weight", prefix)] = attention.Value.W.Value()
			params[fmt.Sprintf("%s.v_proj.bias", prefix)] = attention.Value.B.Value()
		}
		params[fmt.Sprintf("%s.self_attn.out_proj.weight", prefixBase)] = block1.Attention.OutputMerge.W.Value()
		params[fmt.Sprintf("%s.self_attn.out_proj.bias", prefixBase)] = block1.Attention.OutputMerge.B.Value()
		params[fmt.Sprintf("%s.self_attn_layer_norm.weight", prefixBase)] = block1.Norm.W.Value()
		params[fmt.Sprintf("%s.self_attn_layer_norm.bias", prefixBase)] = block1.Norm.B.Value()

		block2 := resolveFeedForwardBlock(layer.FF)
		params[fmt.Sprintf("%s.fc1.weight", prefixBase)] = block2.FFN[0].(*linear.Model).W.Value()
		params[fmt.Sprintf("%s.fc1.bias", prefixBase)] = block2.FFN[0].(*linear.Model).B.Value()
		params[fmt.Sprintf("%s.fc2.weight", prefixBase)] = block2.FFN[2].(*linear.Model).W.Value()
		params[fmt.Sprintf("%s.fc2.bias", prefixBase)] = block2.FFN[2].(*linear.Model).B.Value()
		params[fmt.Sprintf("%s.final_layer_norm.weight", prefixBase)] = block2.Norm.W.Value()
		params[fmt.Sprintf("%s.final_layer_norm.bias", prefixBase)] = block2.Norm.B.Value()
	}

	params["model.encoder.layernorm_embedding.weight"] = encoder.Embeddings.Norm.W.Value()
	params["model.encoder.layernorm_embedding.bias"] = encoder.Embeddings.Norm.B.Value()
	params["model.encoder.layer_norm.weight"] = encoder.LayerNorm.W.Value()
	params["model.encoder.layer_norm.bias"] = encoder.LayerNorm.B.Value()
}

// mapDecoderParams maps the decoder parameters.
func mapDecoderParams(decoder *bart.Decoder, params paramsMap) {
	for i := 0; i < decoder.Config.DecoderLayers; i++ {
		layer := decoder.Layers[i]
		prefixBase := fmt.Sprintf("model.decoder.layers.%d", i)

		block := resolveSelfAttentionBlock(layer.SelfAttention)
		for j := 0; j < decoder.Config.DecoderAttentionHeads; j++ {
			attention := block.Attention.Heads[j]
			prefix := fmt.Sprintf("%s.%d.self_attn", prefixBase, j)
			params[fmt.Sprintf("%s.q_proj.weight", prefix)] = attention.Query.W.Value()
			params[fmt.Sprintf("%s.q_proj.bias", prefix)] = attention.Query.B.Value()
			params[fmt.Sprintf("%s.k_proj.weight", prefix)] = attention.Key.W.Value()
			params[fmt.Sprintf("%s.k_proj.bias", prefix)] = attention.Key.B.Value()
			params[fmt.Sprintf("%s.v_proj.weight", prefix)] = attention.Value.W.Value()
			params[fmt.Sprintf("%s.v_proj.bias", prefix)] = attention.Value.B.Value()
		}
		params[fmt.Sprintf("%s.self_attn.out_proj.weight", prefixBase)] = block.Attention.OutputMerge.W.Value()
		params[fmt.Sprintf("%s.self_attn.out_proj.bias", prefixBase)] = block.Attention.OutputMerge.B.Value()
		params[fmt.Sprintf("%s.self_attn_layer_norm.weight", prefixBase)] = block.Norm.W.Value()
		params[fmt.Sprintf("%s.self_attn_layer_norm.bias", prefixBase)] = block.Norm.B.Value()

		block2 := resolveCrossAttentionBlock(layer.CrossAttention)
		for j := 0; j < decoder.Config.DecoderAttentionHeads; j++ {
			attention := block2.Attention.Heads[j]
			prefix := fmt.Sprintf("%s.%d.encoder_attn", prefixBase, j)
			params[fmt.Sprintf("%s.q_proj.weight", prefix)] = attention.Query.W.Value()
			params[fmt.Sprintf("%s.q_proj.bias", prefix)] = attention.Query.B.Value()
			params[fmt.Sprintf("%s.k_proj.weight", prefix)] = attention.Key.W.Value()
			params[fmt.Sprintf("%s.k_proj.bias", prefix)] = attention.Key.B.Value()
			params[fmt.Sprintf("%s.v_proj.weight", prefix)] = attention.Value.W.Value()
			params[fmt.Sprintf("%s.v_proj.bias", prefix)] = attention.Value.B.Value()
		}
		params[fmt.Sprintf("%s.encoder_attn.out_proj.weight", prefixBase)] = block2.Attention.OutputMerge.W.Value()
		params[fmt.Sprintf("%s.encoder_attn.out_proj.bias", prefixBase)] = block2.Attention.OutputMerge.B.Value()
		params[fmt.Sprintf("%s.encoder_attn_layer_norm.weight", prefixBase)] = block2.Norm.W.Value()
		params[fmt.Sprintf("%s.encoder_attn_layer_norm.bias", prefixBase)] = block2.Norm.B.Value()

		block3 := resolveFeedForwardBlock(layer.FF)
		params[fmt.Sprintf("%s.fc1.weight", prefixBase)] = block3.FFN[0].(*linear.Model).W.Value()
		params[fmt.Sprintf("%s.fc1.bias", prefixBase)] = block3.FFN[0].(*linear.Model).B.Value()
		params[fmt.Sprintf("%s.fc2.weight", prefixBase)] = block3.FFN[2].(*linear.Model).W.Value()
		params[fmt.Sprintf("%s.fc2.bias", prefixBase)] = block3.FFN[2].(*linear.Model).B.Value()
		params[fmt.Sprintf("%s.final_layer_norm.weight", prefixBase)] = block3.Norm.W.Value()
		params[fmt.Sprintf("%s.final_layer_norm.bias", prefixBase)] = block3.Norm.B.Value()
	}

	params["model.decoder.layernorm_embedding.weight"] = decoder.Embeddings.Norm.W.Value()
	params["model.decoder.layernorm_embedding.bias"] = decoder.Embeddings.Norm.B.Value()
	params["model.decoder.layer_norm.weight"] = decoder.LayerNorm.W.Value()
	params["model.decoder.layer_norm.bias"] = decoder.LayerNorm.B.Value()
}

// resolveSelfAttentionBlock resolves the self attention block of the given layer.
func resolveSelfAttentionBlock(m bart.ResidualNormSelfAttention) *bart.SelfAttentionBlock {
	switch m := m.(type) {
	case bart.PreNormSelfAttentionBlock:
		return m.SelfAttentionBlock
	case bart.PostNormSelfAttentionBlock:
		return m.SelfAttentionBlock
	default:
		panic("unknown model")
	}
}

// resolveCrossAttentionBlock resolves the cross attention block of the given layer.
func resolveCrossAttentionBlock(m bart.ResidualNormCrossAttention) *bart.CrossAttentionBlock {
	switch m := m.(type) {
	case bart.PreNormCrossAttentionBlock:
		return m.CrossAttentionBlock
	case bart.PostNormCrossAttentionBlock:
		return m.CrossAttentionBlock
	default:
		panic("unknown model")
	}
}

// forwardBlock resolves the forward block of the given layer.
func resolveFeedForwardBlock(m bart.ResidualNormFeedForward) *bart.FeedForwardBlock {
	switch m := m.(type) {
	case bart.PreNormFeedForwardBlock:
		return m.FeedForwardBlock
	case bart.PostNormFeedForwardBlock:
		return m.FeedForwardBlock
	default:
		panic("unknown model")
	}
}
