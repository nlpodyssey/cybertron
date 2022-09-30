// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"context"
	"fmt"
	"path"
	"path/filepath"
	"sort"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks/languagemodeling"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/cybertron/pkg/vocabulary"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/spago/nn"
)

// LanguageModel is a masked language model.
type LanguageModel struct {
	// Model is the model used to answer questions.
	Model *bert.ModelForMaskedLM
	// Words vocabulary
	vocab *vocabulary.Vocabulary
	// Tokenizer is the tokenizer used to tokenize questions and passages.
	Tokenizer *wordpiecetokenizer.WordPieceTokenizer
	// doLowerCase is a flag indicating if the model should lowercase the input before tokenization.
	doLowerCase bool
	// embeddingsRepo is the repository used for loading embeddings.
	embeddingsRepo *diskstore.Repository
}

// LoadMaskedLanguageModel returns a LanguageModel loading the model, the embeddings and the tokenizer from a directory.
func LoadMaskedLanguageModel(modelPath string) (*LanguageModel, error) {
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

	m, err := nn.LoadFromFile[*bert.ModelForMaskedLM](path.Join(modelPath, "spago_model.bin"))
	if err != nil {
		return nil, fmt.Errorf("failed to load bart model: %w", err)
	}

	err = m.Bert.SetEmbeddings(embeddingsRepo)
	if err != nil {
		return nil, fmt.Errorf("failed to set embeddings: %w", err)
	}

	return &LanguageModel{
		Model:          m,
		vocab:          vocab,
		Tokenizer:      tokenizer,
		doLowerCase:    tokenizerConfig.DoLowerCase,
		embeddingsRepo: embeddingsRepo,
	}, nil
}

// Predict returns the predicted tokens
func (m *LanguageModel) Predict(_ context.Context, text string) (languagemodeling.Response, error) {
	tokenized := pad(m.tokenize(text))
	prediction := m.Model.Predict(tokenizers.GetStrings(tokenized))

	result := make([]languagemodeling.Token, 0, len(prediction))
	for i, logits := range prediction {
		probs := logits.Value().Softmax()
		argmax := probs.ArgMax()
		score := probs.AtVec(argmax).Scalar().F64()
		word, ok := m.vocab.Term(argmax)
		if !ok {
			word = wordpiecetokenizer.DefaultUnknownToken // if this is returned, there's a misalignment with the vocabulary
		}
		start, end := tokenized[i].Offsets.Start, tokenized[i].Offsets.End
		result = append(result, languagemodeling.Token{
			Text:  word,
			Start: start,
			End:   end,
			Score: score,
		})
	}

	sort.SliceStable(result, func(i, j int) bool {
		return result[i].Start > result[i].End
	})

	return languagemodeling.Response{
		Tokens: result,
	}, nil
}

// tokenize returns the tokens of the given text (without padding tokens).
func (m *LanguageModel) tokenize(text string) []tokenizers.StringOffsetsPair {
	if m.doLowerCase {
		text = strings.ToLower(text)
	}
	return m.Tokenizer.Tokenize(text)
}

func pad(tokens []tokenizers.StringOffsetsPair) []tokenizers.StringOffsetsPair {
	return append(prepend(tokens, tokenizers.StringOffsetsPair{String: wordpiecetokenizer.DefaultClassToken}),
		tokenizers.StringOffsetsPair{String: wordpiecetokenizer.DefaultSequenceSeparator})
}

func prepend(x []tokenizers.StringOffsetsPair, y tokenizers.StringOffsetsPair) []tokenizers.StringOffsetsPair {
	return append([]tokenizers.StringOffsetsPair{y}, x...)
}
