// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"fmt"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textclassification"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/cybertron/pkg/utils/sliceutils"
	"github.com/nlpodyssey/cybertron/pkg/vocabulary"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/spago/nn"
)

// TextClassification is a text classification model.
type TextClassification struct {
	// Model is the model used to answer questions.
	Model *bert.ModelForSequenceClassification
	// Tokenizer is the tokenizer used to tokenize questions and passages.
	Tokenizer *wordpiecetokenizer.WordPieceTokenizer
	// doLowerCase is a flag indicating if the model should lowercase the input before tokenization.
	doLowerCase bool
	// embeddingsRepo is the repository used for loading embeddings.
	embeddingsRepo *diskstore.Repository
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

	embeddingsRepo, err := diskstore.NewRepository(filepath.Join(modelPath, "repo"), diskstore.ReadOnlyMode)
	if err != nil {
		return nil, fmt.Errorf("failed to load embeddings repository for text classification: %w", err)
	}

	m, err := nn.LoadFromFile[*bert.ModelForSequenceClassification](path.Join(modelPath, "spago_model.bin"))
	if err != nil {
		return nil, fmt.Errorf("failed to load bart model: %w", err)
	}

	err = m.Bert.SetEmbeddings(embeddingsRepo)
	if err != nil {
		return nil, fmt.Errorf("failed to set embeddings: %w", err)
	}

	return &TextClassification{
		Model:          m,
		Tokenizer:      tokenizer,
		doLowerCase:    tokenizerConfig.DoLowerCase,
		embeddingsRepo: embeddingsRepo,
	}, nil
}

// Classify returns the classification of the given text.
func (m *TextClassification) Classify(text string) (textclassification.Response, error) {
	logits := m.Model.Classify(m.tokenize(text))
	defer func() {
		go ag.ReleaseGraph(logits)
	}()

	probs := logits.Value().Softmax()

	result := sliceutils.NewIndexedSlice[float64](probs.Data().F64())
	sort.Stable(sort.Reverse(result))

	response := textclassification.Response{
		Labels: result.Indices,
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
