// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config contains the global configuration of the Bart model and the heads of fine-tuning tasks.
// The configuration coincides with that of Hugging Face to facilitate compatibility between the two architectures.
type Config struct {
	NumLabels                  int               `json:"_num_labels,omitempty"`
	ActivationDropout          float64           `json:"activation_dropout,omitempty"`
	ActivationFunction         string            `json:"activation_function,omitempty"`
	BiasLogits                 bool              `json:"add_bias_logits,omitempty"`
	FinalLayerNorm             bool              `json:"add_final_layer_norm,omitempty"`
	Architecture               []string          `json:"architectures,omitempty"`
	AttentionDropout           float64           `json:"attention_dropout,omitempty"`
	BosTokenID                 int               `json:"bos_token_id,omitempty"`
	ClassifierDropout          float64           `json:"classif_dropout,omitempty"`
	DModel                     int               `json:"d_model,omitempty"`
	DecoderAttentionHeads      int               `json:"decoder_attention_heads,omitempty"`
	DecoderFFNDim              int               `json:"decoder_ffn_dim,omitempty"`
	DecoderLayerDrop           float64           `json:"decoder_layerdrop,omitempty"`
	DecoderLayers              int               `json:"decoder_layers,omitempty"`
	DecoderStartTokenID        int               `json:"decoder_start_token_id,omitempty"`
	Dropout                    float64           `json:"dropout,omitempty"`
	EncoderAttentionHeads      int               `json:"encoder_attention_heads,omitempty"`
	EncoderFFNDim              int               `json:"encoder_ffn_dim,omitempty"`
	EncoderLayerDrop           float64           `json:"encoder_layerdrop,omitempty"`
	EncoderLayers              int               `json:"encoder_layers,omitempty"`
	EosTokenID                 int               `json:"eos_token_id,omitempty"`
	FineTuningTask             string            `json:"finetuning_task,omitempty"`
	ForceBosTokenToBeGenerated bool              `json:"force_bos_token_to_be_generated,omitempty"`
	ID2Label                   map[string]string `json:"id2label,omitempty"`
	InitStd                    float64           `json:"init_std,omitempty"`
	IsEncoderDecoder           bool              `json:"is_encoder_decoder,omitempty"`
	Label2ID                   map[string]int    `json:"label2id,omitempty"`
	LengthPenalty              float64           `json:"length_penalty,omitempty"`
	MaxPositionEmbeddings      int               `json:"max_position_embeddings,omitempty"`
	ModelType                  string            `json:"model_type,omitempty"`
	NormalizeBefore            bool              `json:"normalize_before,omitempty"`
	NormalizeEmbedding         bool              `json:"normalize_embedding,omitempty"`
	NumHiddenLayers            int               `json:"num_hidden_layers,omitempty"`
	OutputPast                 bool              `json:"output_past,omitempty"`
	PadTokenID                 int               `json:"pad_token_id,omitempty"`
	ScaleEmbedding             bool              `json:"scale_embedding,omitempty"`
	StaticPositionEmbeddings   bool              `json:"static_position_embeddings,omitempty"`
	TotalFlos                  float64           `json:"total_flos,omitempty"`
	VocabSize                  int               `json:"vocab_size,omitempty"`
	NumBeams                   int               `json:"num_beams,omitempty"`
	MaxLength                  int               `json:"max_length,omitempty"`
	MinLength                  int               `json:"min_length,omitempty"`
	BadWordsIDs                [][]int           `json:"bad_words_ids,omitempty"`
	EarlyStopping              bool              `json:"early_stopping,omitempty"`
	NoRepeatNGramSize          int               `json:"no_repeat_ngram_size,omitempty"`
	ExtraSpecialTokens         map[int]string    `json:"extra_special_tokens,omitempty"`
	Cybertron                  struct {
		Training                           bool   `json:"training,omitempty"`
		PositionalEncoderOffset            int    `json:"positional_encoder_offset,omitempty"`
		SharedEmbeddingsStoreName          string `json:"shared_embeddings_store_name,omitempty"`
		DecoderPositionalEncodingStoreName string `json:"decoder_positional_encoding_store_name,omitempty"`
		EncoderPositionalEncodingStoreName string `json:"encoder_positional_encoding_store_name,omitempty"`
	}
}

// ConfigFromFile loads a Bart model Config from file.
func ConfigFromFile(file string) (Config, error) {
	config := baseConfig()
	configFile, err := os.Open(file)
	if err != nil {
		return Config{}, err
	}
	defer configFile.Close()
	err = json.NewDecoder(configFile).Decode(&config)
	if err != nil {
		return Config{}, err
	}

	// Set default values
	if config.MaxLength == 0 {
		config.MaxLength = config.MaxPositionEmbeddings
	}
	return config, nil
}

func baseConfig() Config {
	return Config{
		NormalizeEmbedding: true,
	}
}

// EntailmentID returns the id of the `entailment` labels.
func (c *Config) EntailmentID() (int, error) {
	id, ok := c.Label2ID["entailment"]
	if !ok {
		return -1, fmt.Errorf("bart: `entailment` label not found")
	}
	return id, nil
}

// ContradictionID returns the id of the `contradiction` labels.
func (c *Config) ContradictionID() (int, error) {
	id, ok := c.Label2ID["contradiction"]
	if !ok {
		return -1, fmt.Errorf("bart: `contradiction` label not found")
	}
	return id, nil
}
