// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"fmt"
	"github.com/nlpodyssey/cybertron/pkg/generationutils"
	"github.com/nlpodyssey/cybertron/pkg/models/bart"
	"github.com/nlpodyssey/cybertron/pkg/tasks/text2text"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers/sentencepiece"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
	"math"
	"path"
	"path/filepath"
)

var _ text2text.Interface = &Text2Text{}

// Text2Text contains the ModelForConditionalGeneration and the Tokenizer
// used for conditional generation tasks.
// For example, Machine Translation and Summarization.
type Text2Text struct {
	// Model is the model used for conditional generation.
	Model *bart.ModelForConditionalGeneration
	// Tokenizer is the tokenizer used for conditional generation.
	Tokenizer *sentencepiece.Tokenizer
	// embeddingsRepo is the repository used for loading embeddings.
	embeddingsRepo *diskstore.Repository
}

// LoadText2Text returns a Text2Text loading the model, the embeddings and the tokenizer from a directory.
func LoadText2Text(modelPath string) (*Text2Text, error) {
	tok, err := sentencepiece.NewFromModelFolder(modelPath, false)
	if err != nil {
		return nil, fmt.Errorf("failed to load sentencepiece tokenizer for text2text: %w", err)
	}

	embeddingsRepo, err := diskstore.NewRepository(filepath.Join(modelPath, "repo"), diskstore.ReadOnlyMode)
	if err != nil {
		return nil, fmt.Errorf("failed to load embeddings repository for text2text: %w", err)
	}

	m, err := nn.LoadFromFile[*bart.ModelForConditionalGeneration](path.Join(modelPath, "spago_model.bin"))
	if err != nil {
		return nil, fmt.Errorf("failed to load bart model: %w", err)
	}

	err = m.Bart.SetEmbeddings(embeddingsRepo)
	if err != nil {
		return nil, fmt.Errorf("failed to load embeddings: %w", err)
	}

	return &Text2Text{
		Model:          m,
		Tokenizer:      tok,
		embeddingsRepo: embeddingsRepo,
	}, nil
}

// Close finalizes the Text2Text resources.
// It satisfies the interface io.Closer.
func (m *Text2Text) Close() error {
	return m.embeddingsRepo.Close()
}

// Generate generates a text from the input.
func (m *Text2Text) Generate(text string, opts text2text.Options) (text2text.Response, error) {
	sequences, scores := m.process(m.Tokenize(text), opts)
	result := text2text.Response{
		Texts:  make([]string, len(sequences)),
		Scores: make([]float64, len(scores)),
	}
	for i, sequence := range sequences {
		result.Texts[i], result.Scores[i] = m.Detokenize(sequence), scores[i]
	}
	return result, nil
}

func (m *Text2Text) process(inputIDs []int, opts text2text.Options) ([][]int, []float64) {
	nodesToRelease := make([]ag.Node, 0)
	defer func() {
		go ag.ReleaseGraph(nodesToRelease...)
	}()

	next := m.Model.DecodingFunc(inputIDs, m.logProbProcessor(opts), true)
	cache := make([]bart.Cache, m.Model.Bart.Config.NumBeams)

	predictNext := func(decodingInputIDs [][]int, lastBeamIndices []int) []mat.Matrix {
		cache = reorderCache(cache, lastBeamIndices)
		batch := m.batch(decodingInputIDs, cache)
		logProbValues := make([]mat.Matrix, len(batch))

		for i, result := range next(batch) {
			logProbValues[i], cache[i] = result.LogProbValue, result.NextCache
			nodesToRelease = append(nodesToRelease, result.LogProbRaw)
		}
		return logProbValues
	}

	decoder := &generationutils.BeamSearchDecoder{
		Config:      decoderConfig(m.Model.Bart.Config),
		PredictNext: predictNext,
		SelectNext:  decodingStrategy(opts),
	}
	return decoder.Decode()
}

// reorderCache reorders the cache according to the last beam indices.
func reorderCache(cache []bart.Cache, lastBeamIndices []int) []bart.Cache {
	tmpCache := make([]bart.Cache, len(cache))
	for i, beamIndex := range lastBeamIndices {
		tmpCache[i] = cache[beamIndex]
	}
	return tmpCache
}

func (m *Text2Text) batch(sequences [][]int, cache []bart.Cache) []*bart.DecodingInput {
	batch := make([]*bart.DecodingInput, len(sequences))
	for i, sequence := range sequences {
		batch[i] = &bart.DecodingInput{
			InputIDs: sequence[len(sequence)-1:],
			Cache:    cache[i],
			CurLen:   len(sequence),
		}
	}
	return batch
}

func decodingStrategy(opts text2text.Options) generationutils.DecodingStrategyFunc {
	if opts.Sample.Valid && opts.Sample.Value {
		return generationutils.SelectNextMultinomial
	}
	return generationutils.SelectNextTopK
}

// logProbProcessor returns a function that processes the log-probabilities.
func (m *Text2Text) logProbProcessor(opts text2text.Options) generationutils.ScoreProcessor {
	procs := make([]generationutils.ScoreProcessor, 0, 3)
	if opts.Temperature.Valid {
		procs = append(procs, generationutils.TemperatureProcessor(opts.Temperature.Value))
	}
	if opts.TopK.Valid {
		procs = append(procs, generationutils.TopKProcessor(opts.TopK.Value, math.Inf(-1)))
	}
	if opts.TopP.Valid {
		minSize := 1
		if m.Model.Bart.Config.NumBeams > 1 {
			minSize = 2
		}
		procs = append(procs, generationutils.TopPProcessor(opts.TopP.Value, math.Inf(-1), minSize))
	}
	return generationutils.ProcessScores(procs...)
}
