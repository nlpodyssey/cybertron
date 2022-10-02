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
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks/languagemodeling"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/cybertron/pkg/vocabulary"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/spago/nn"
)

const defaultTopK = 10

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
func (m *LanguageModel) Predict(_ context.Context, text string, parameters languagemodeling.Parameters) (languagemodeling.Response, error) {
	if parameters.K == 0 {
		parameters.K = defaultTopK
	}

	tokenized := pad(m.tokenize(text))
	prediction := m.Model.Predict(tokenizers.GetStrings(tokenized))

	result := make([]languagemodeling.Token, 0, len(prediction))
	for i, logits := range prediction {
		probs := logits.Value().Softmax()

		scores := make([]float64, 0)
		words := make([]string, 0)
		for _, item := range selectTopK(probs, parameters.K) {
			word, ok := m.vocab.Term(item.Index)
			if !ok {
				word = wordpiecetokenizer.DefaultUnknownToken // if this is returned, there's a misalignment with the vocabulary
			}
			words = append(words, word)
			scores = append(scores, item.Score)
		}

		start, end := tokenized[i].Offsets.Start, tokenized[i].Offsets.End
		result = append(result, languagemodeling.Token{
			Start:  start,
			End:    end,
			Words:  words,
			Scores: scores,
		})
	}

	sort.SliceStable(result, func(i, j int) bool {
		return result[i].Start < result[j].Start
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

type IndexScorePair struct {
	Index int
	Score float64
}

// selectTopK returns the next tokens to be generated.
func selectTopK(scores mat.Matrix, resultSize int) []*IndexScorePair {
	if resultSize == 1 {
		argmax := scores.ArgMax()
		return []*IndexScorePair{
			{
				Index: argmax,
				Score: scores.ScalarAtVec(argmax).F64(),
			},
		}
	}

	arena := make([]IndexScorePair, resultSize)
	result := make([]*IndexScorePair, 0, resultSize)

	var minScore float64
	minIndex := -1

	for i, score := range scores.Data().F64() {

		if len(result) < resultSize {
			if minIndex == -1 || score < minScore {
				minScore = score
				minIndex = len(result)
			}

			st := &arena[0]
			arena = arena[1:]

			st.Index = i
			st.Score = score

			result = append(result, st)
			continue
		}

		if score <= minScore {
			continue
		}

		// Replace the scored token with minimum score with the new one
		st := result[minIndex]
		st.Index = i
		st.Score = score

		// Find the new minimum
		minScore = result[0].Score
		minIndex = 0
		for j, v := range result {
			if v.Score < minScore {
				minScore = v.Score
				minIndex = j
			}
		}
	}

	sort.SliceStable(result, func(i, j int) bool {
		return result[i].Score > result[j].Score
	})

	return result
}
