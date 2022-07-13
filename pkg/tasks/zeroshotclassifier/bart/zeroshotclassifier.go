// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"context"
	"fmt"
	"path"
	"path/filepath"
	"runtime"
	"sort"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/models/bart"
	"github.com/nlpodyssey/cybertron/pkg/tasks/zeroshotclassifier"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers/bpetokenizer"
	"github.com/nlpodyssey/cybertron/pkg/utils/sliceutils"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"golang.org/x/sync/errgroup"
)

var _ zeroshotclassifier.Interface = &ZeroShotClassifier{}

const (
	defaultStartTokenID = 0
	defaultEndTokenID   = 2
)

// ZeroShotClassifier contains the ModelForSequenceClassification and the Tokenizer
// used for zero-shot classification tasks.
type ZeroShotClassifier struct {
	// Model is the model used for zero-shot classification.
	Model *bart.ModelForSequenceClassification
	// Tokenizer is the tokenizer.
	Tokenizer                     *bpetokenizer.BPETokenizer
	embeddingsRepo                *diskstore.Repository
	entailmentID, contradictionID int
}

// LoadZeroShotClassifier loads a ZeroShotClassifier from a directory.
func LoadZeroShotClassifier(modelPath string) (*ZeroShotClassifier, error) {
	tok, err := bpetokenizer.NewFromModelFolder(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load sentencepiece tokenizer for zero-shot: %w", err)
	}

	embeddingsRepo, err := diskstore.NewRepository(filepath.Join(modelPath, "repo"), diskstore.ReadOnlyMode)
	if err != nil {
		return nil, fmt.Errorf("failed to load embeddings repository for zero-shot: %w", err)
	}

	m, err := nn.LoadFromFile[*bart.ModelForSequenceClassification](path.Join(modelPath, "spago_model.bin"))
	if err != nil {
		return nil, fmt.Errorf("failed to load bart model: %w", err)
	}

	err = m.Bart.SetEmbeddings(embeddingsRepo)
	if err != nil {
		return nil, fmt.Errorf("failed to load embeddings: %w", err)
	}

	entailmentID, err := m.Bart.Config.EntailmentID()
	if err != nil {
		return nil, err
	}
	contradictionID, err := m.Bart.Config.ContradictionID()
	if err != nil {
		return nil, err
	}

	return &ZeroShotClassifier{
		Model:           m,
		Tokenizer:       tok,
		embeddingsRepo:  embeddingsRepo,
		entailmentID:    entailmentID,
		contradictionID: contradictionID,
	}, nil
}

// Close finalizes the ZeroShotClassifier resources.
// It satisfies the interface io.Closer.
func (m *ZeroShotClassifier) Close() error {
	return m.embeddingsRepo.Close()
}

// Classify classifies the input.
func (m *ZeroShotClassifier) Classify(_ context.Context, text string, parameters zeroshotclassifier.Parameters) (zeroshotclassifier.Response, error) {
	premise, err := m.tokenize(text, defaultStartTokenID, defaultEndTokenID)
	if err != nil {
		return zeroshotclassifier.Response{}, err
	}

	multiClass := parameters.MultiLabel || len(parameters.CandidateLabels) == 1
	scoreFn := m.score(premise, multiClass)

	ch := make(chan struct{}, runtime.NumCPU())
	eg, _ := errgroup.WithContext(context.Background())

	var scores mat.Matrix = mat.NewEmptyVecDense[float64](len(parameters.CandidateLabels))

	for i := range parameters.CandidateLabels {
		ch <- struct{}{}
		i := i
		eg.Go(func() error {
			hypothesis, err := m.tokenize(
				strings.Replace(parameters.HypothesisTemplate, "{}", parameters.CandidateLabels[i], -1),
				defaultEndTokenID,
				defaultEndTokenID,
			)
			if err == nil {
				score := scoreFn(hypothesis)
				scores.SetVecScalar(i, float.Interface(score))
			}
			<-ch
			return err
		})
	}
	if err := eg.Wait(); err != nil {
		return zeroshotclassifier.Response{}, err
	}
	for i := 0; i < len(ch); i++ {
		ch <- struct{}{}
	}
	close(ch)

	if !multiClass {
		scores = scores.Softmax() // softmax the "entailment" over all candidate labels
	}

	result := sliceutils.NewIndexedSlice[float64](scores.Data().F64())
	sort.Stable(sort.Reverse(result))

	labels := make([]string, len(parameters.CandidateLabels))
	for i, ii := range result.Indices {
		labels[i] = parameters.CandidateLabels[ii]
	}

	response := zeroshotclassifier.Response{
		Labels: labels,
		Scores: result.Slice,
	}
	return response, nil
}
