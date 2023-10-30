// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"context"
	"fmt"
	"github.com/nlpodyssey/spago/mat"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textclassification"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/cybertron/pkg/utils/sliceutils"
	"github.com/nlpodyssey/cybertron/pkg/vocabulary"
	"github.com/nlpodyssey/spago/nn"
	"github.com/rs/zerolog/log"
)

// TextClassification is a text classification model.
type TextClassification struct {
	// Model is the model used to answer questions.
	Model *bert.ModelForSequenceClassification
	// Tokenizer is the tokenizer used to tokenize questions and passages.
	Tokenizer *wordpiecetokenizer.WordPieceTokenizer
	// Labels is the list of labels used for classification.
	Labels []string
	// doLowerCase is a flag indicating if the model should lowercase the input before tokenization.
	doLowerCase bool
}

// LoadTextClassification returns a TextClassification loading the model, the embeddings and the tokenizer from a directory.
func LoadTextClassification(modelPath string) (*TextClassification, error) {
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

	m, err := nn.LoadFromFile[*bert.ModelForSequenceClassification](path.Join(modelPath, "spago_model.bin"))
	if err != nil {
		return nil, fmt.Errorf("failed to load bart model: %w", err)
	}

	return &TextClassification{
		Model:       m,
		Tokenizer:   tokenizer,
		Labels:      labels,
		doLowerCase: tokenizerConfig.DoLowerCase,
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
func (m *TextClassification) Classify(_ context.Context, text string) (textclassification.Response, error) {
	tokenized := m.tokenize(text)
	if l, k := len(tokenized), m.Model.Bert.Config.MaxPositionEmbeddings; l > k {
		return textclassification.Response{}, fmt.Errorf("%w: %d > %d", textclassification.ErrInputSequenceTooLong, l, k)
	}
	logits := m.Model.Classify(tokenized)
	probs := logits.Value().(mat.Matrix).Softmax()

	result := sliceutils.NewIndexedSlice[float64](probs.Data().F64())
	sort.Stable(sort.Reverse(result))

	labels := make([]string, len(m.Labels))
	for i, ii := range result.Indices {
		labels[i] = m.Labels[ii]
	}

	response := textclassification.Response{
		Labels: labels,
		Scores: result.Slice,
	}
	return response, nil
}

// tokenize returns the tokens of the given text (including padding tokens).
func (m *TextClassification) tokenize(text string) []string {
	if m.doLowerCase {
		text = strings.ToLower(text)
	}
	cls := wordpiecetokenizer.DefaultClassToken
	sep := wordpiecetokenizer.DefaultSequenceSeparator
	return append([]string{cls}, append(tokenizers.GetStrings(m.Tokenizer.Tokenize(text)), sep)...)
}
