// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"context"
	"fmt"
	"path"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks/tokenclassification"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/cybertron/pkg/vocabulary"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/spago/nn"
	"github.com/rs/zerolog/log"
)

// TokenClassification is a token classification model.
type TokenClassification struct {
	// Model is the model used to answer questions.
	Model *ModelForTokenClassification
	// Tokenizer is the tokenizer used to tokenize questions and passages.
	Tokenizer *wordpiecetokenizer.WordPieceTokenizer
	// Labels is the list of labels used for classification.
	Labels []string
	// doLowerCase is a flag indicating if the model should lowercase the input before tokenization.
	doLowerCase bool
	// embeddingsRepo is the repository used for loading embeddings.
	embeddingsRepo *diskstore.Repository
}

// LoadTokenClassification returns a TokenClassification loading the model, the embeddings and the tokenizer from a directory.
func LoadTokenClassification(modelPath string) (*TokenClassification, error) {
	vocab, err := vocabulary.NewFromFile(filepath.Join(modelPath, "vocab.txt"))
	if err != nil {
		return nil, fmt.Errorf("failed to load vocabulary for text classification: %w", err)
	}
	tokenizer := wordpiecetokenizer.New(vocab)

	tokenizerConfig, err := bert.ConfigFromFile[bert.TokenizerConfig](path.Join(modelPath, "tokenizer_config.json"))
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer config for text classification: %w", err)
	}

	config, err := bert.ConfigFromFile[bert.Config](path.Join(modelPath, "config.json"))
	if err != nil {
		return nil, fmt.Errorf("failed to load config for text classification: %w", err)
	}
	labels := ID2Label(config.ID2Label)

	embeddingsRepo, err := diskstore.NewRepository(filepath.Join(modelPath, "repo"), diskstore.ReadOnlyMode)
	if err != nil {
		return nil, fmt.Errorf("failed to load embeddings repository for text classification: %w", err)
	}

	m, err := nn.LoadFromFile[*bert.ModelForTokenClassification](path.Join(modelPath, "spago_model.bin"))
	if err != nil {
		return nil, fmt.Errorf("failed to load bart model: %w", err)
	}

	err = m.Bert.SetEmbeddings(embeddingsRepo)
	if err != nil {
		return nil, fmt.Errorf("failed to set embeddings: %w", err)
	}

	return &TokenClassification{
		Model:          &ModelForTokenClassification{ModelForTokenClassification: m},
		Tokenizer:      tokenizer,
		Labels:         labels,
		doLowerCase:    tokenizerConfig.DoLowerCase,
		embeddingsRepo: embeddingsRepo,
	}, nil
}

func ID2Label(value map[string]string) []string {
	if len(value) == 0 {
		return []string{"LABEL_0", "LABEL_1"} // assume binary classification by default
	}
	y := make([]string, len(value))
	for k, v := range value {
		i, err := strconv.Atoi(k)
		if err != nil {
			log.Fatal().Err(err).Send()
		}
		y[i] = v
	}
	return y
}

// Classify returns the classification of the given text.
func (m *TokenClassification) Classify(_ context.Context, text string, parameters tokenclassification.Parameters) (tokenclassification.Response, error) {
	tokenized := m.tokenize(text)
	if l, max := len(tokenized), m.Model.Bert.Config.MaxPositionEmbeddings; l > max {
		return tokenclassification.Response{}, fmt.Errorf("%w: %d > %d", tokenclassification.ErrInputSequenceTooLong, l, max)
	}

	logits := m.Model.Classify(pad(tokenizers.GetStrings(tokenized)))
	tokens := make([]tokenclassification.Token, 0, len(tokenized))
	for i, token := range wordpiecetokenizer.GroupSubWords(tokenized) {
		label, score := m.getBestClass(logits[i])

		tokens = append(tokens, tokenclassification.Token{
			Text:  text[token.Offsets.Start:token.Offsets.End],
			Start: token.Offsets.Start,
			End:   token.Offsets.End,
			Label: label,
			Score: score,
		})
	}

	if parameters.AggregationStrategy == tokenclassification.AggregationStrategySimple {
		tokens = tokenclassification.FilterNotEntities(tokenclassification.Aggregate(tokens))
	}

	response := tokenclassification.Response{
		Tokens: tokens,
	}
	return response, nil
}

func (m *TokenClassification) getBestClass(logits ag.Node) (label string, score float64) {
	probs := logits.Value().Softmax()
	argmax := probs.ArgMax()
	score = probs.AtVec(argmax).Scalar().F64()
	label = m.Labels[argmax]
	return
}

// tokenize returns the tokens of the given text (without padding tokens).
func (m *TokenClassification) tokenize(text string) []tokenizers.StringOffsetsPair {
	if m.doLowerCase {
		text = strings.ToLower(text)
	}
	return m.Tokenizer.Tokenize(text)
}

func pad(tokens []string) []string {
	return append(prepend(tokens, wordpiecetokenizer.DefaultClassToken), wordpiecetokenizer.DefaultSequenceSeparator)
}

func prepend(x []string, y string) []string {
	return append([]string{y}, x...)
}
