// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"encoding/json"
	"os"
)

// Config contains the global configuration of the Bert model and the heads of fine-tuning tasks.
// The configuration coincides with that of Hugging Face to facilitate compatibility between the two architectures.
type Config struct {
	Architectures             []string          `json:"architectures"`
	AttentionProbsDropoutProb float64           `json:"attention_probs_dropout_prob"`
	GradientCheckpointing     bool              `json:"gradient_checkpointing"`
	HiddenAct                 string            `json:"hidden_act"`
	HiddenDropoutProb         float64           `json:"hidden_dropout_prob"`
	HiddenSize                int               `json:"hidden_size"`
	EmbeddingsSize            int               `json:"embeddings_size"`
	InitializerRange          float64           `json:"initializer_range"`
	IntermediateSize          int               `json:"intermediate_size"`
	LayerNormEps              float64           `json:"layer_norm_eps"`
	MaxPositionEmbeddings     int               `json:"max_position_embeddings"`
	ModelType                 string            `json:"model_type"`
	NumAttentionHeads         int               `json:"num_attention_heads"`
	NumHiddenLayers           int               `json:"num_hidden_layers"`
	PadTokenId                int               `json:"pad_token_id"`
	PositionEmbeddingType     string            `json:"position_embedding_type"`
	TransformersVersion       string            `json:"transformers_version"`
	TypeVocabSize             int               `json:"type_vocab_size"`
	UseCache                  bool              `json:"use_cache"`
	VocabSize                 int               `json:"vocab_size"`
	ID2Label                  map[string]string `json:"id2label"`
	Cybertron                 struct {
		Training            bool   `json:"training"`
		TokensStoreName     string `json:"tokens_store_name"`
		PositionsStoreName  string `json:"positions_store_name"`
		TokenTypesStoreName string `json:"token_types_store_name"`
	}
}

// TokenizerConfig contains the configuration of the tokenizer.
// The configuration coincides with that of Hugging Face to facilitate compatibility between the two architectures.
type TokenizerConfig struct {
	DoLowerCase          bool        `json:"do_lower_case"`
	UnkToken             string      `json:"unk_token"`
	SepToken             string      `json:"sep_token"`
	PadToken             string      `json:"pad_token"`
	ClsToken             string      `json:"cls_token"`
	MaskToken            string      `json:"mask_token"`
	TokenizeChineseChars bool        `json:"tokenize_chinese_chars"`
	StripAccents         interface{} `json:"strip_accents"`
	ModelMaxLength       int         `json:"model_max_length"`
}

// ConfigFile is the union of the configuration structures.
type ConfigFile interface {
	Config | TokenizerConfig
}

// ConfigFromFile loads a Bart model Config from file.
func ConfigFromFile[T ConfigFile](file string) (config T, _ error) {
	configFile, err := os.Open(file)
	if err != nil {
		return config, err
	}
	defer configFile.Close()
	err = json.NewDecoder(configFile).Decode(&config)
	if err != nil {
		return config, err
	}
	return config, nil
}
