package distilbert

import (
	"encoding/json"
	"os"
)

// Config contains the global configuration of the DistilBert model and the heads of fine-tuning tasks.
// The configuration coincides with that of Hugging Face to facilitate compatibility between the two architectures.
type Config struct {
	// Parameters for fine-tuning tasks
	Architectures []string          `json:"architectures"`
	ID2Label      map[string]string `json:"id2label"`
	// DistilBert configuration
	Activation            string  `json:"activation"`
	VocabSize             int     `json:"vocab_size"`
	MaxPositionEmbeddings int     `json:"max_position_embeddings"`
	NumAttentionHeads     int     `json:"n_heads"`
	NumHiddenLayers       int     `json:"n_layers"`
	HiddenSize            int     `json:"dim"`
	IntermediateSize      int     `json:"hidden_dim"`
	InitializerRange      float64 `json:"initializer_range"`
	HiddenDropout         float64 `json:"dropout"`
	AttentionDropout      float64 `json:"attention_dropout"`
	QADropout             float64 `json:"qa_dropout"`
	SeqClsDropout         float64 `json:"seq_classif_dropout"`
	EmbeddingsSize        int     `json:"embeddings_size"`
	//Pretrained configuration
	ModelType  string `json:"model_type"`
	PadTokenId int    `json:"pad_token_id"`
	// Cybertron configuration
	Cybertron struct {
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
