// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"context"
	"fmt"
	"path"
	"runtime"
	"sort"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/models/bart"
	"github.com/nlpodyssey/cybertron/pkg/tasks/zeroshotclassifier"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers/bpetokenizer"
	"github.com/nlpodyssey/cybertron/pkg/utils/sliceutils"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/embedding"
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
	entailmentID, contradictionID int
}

// LoadZeroShotClassifier loads a ZeroShotClassifier from a directory.
func LoadZeroShotClassifier(modelPath string) (*ZeroShotClassifier, error) {
	tok, err := bpetokenizer.NewFromModelFolder(modelPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load sentencepiece tokenizer for zero-shot: %w", err)
	}

	m, err := nn.LoadFromFile[*bart.ModelForSequenceClassification](path.Join(modelPath, "spago_model.bin"))
	if err != nil {
		return nil, fmt.Errorf("failed to load bart model: %w", err)
	}

	m.Bart.Encoder.Embeddings.SharedEmbeddings = embedding.Shared{Model: m.Bart.Embeddings}
	m.Bart.Decoder.Embeddings.SharedEmbeddings = embedding.Shared{Model: m.Bart.Embeddings}

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
		entailmentID:    entailmentID,
		contradictionID: contradictionID,
	}, nil
}

// Classify classifies the input.
func (m *ZeroShotClassifier) Classify(_ context.Context, text string, parameters zeroshotclassifier.Parameters) (zeroshotclassifier.Response, error) {
	premise, err := m.tokenize(text, defaultStartTokenID, defaultEndTokenID)
	if err != nil {
		return zeroshotclassifier.Response{}, err
	}
	if l, k := len(premise), m.Model.Bart.Config.MaxLength; l > k {
		return zeroshotclassifier.Response{}, fmt.Errorf("%w: %d > %d", zeroshotclassifier.ErrInputSequenceTooLong, l, k)
	}

	// If the API request does not specify a HypothesisTemplate, then use the default
	// I believe its more robust to do this here, rather than modifying ZeroShotParameters.GetHypothesisTemplate() PB definition
	if parameters.HypothesisTemplate == "" {
		parameters.HypothesisTemplate = zeroshotclassifier.DefaultHypothesisTemplate
	}

	multiClass := parameters.MultiLabel || len(parameters.CandidateLabels) == 1
	scoreFn := m.score(premise, multiClass)

	ch := make(chan struct{}, runtime.NumCPU())
	eg, _ := errgroup.WithContext(context.Background())

	var scores mat.Matrix = mat.NewDense[float64](mat.WithShape(len(parameters.CandidateLabels)))

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
				scores.SetScalar(float.Interface(score), i)
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
