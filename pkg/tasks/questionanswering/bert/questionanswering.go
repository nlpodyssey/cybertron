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
	"github.com/nlpodyssey/cybertron/pkg/tasks/questionanswering"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/cybertron/pkg/utils/sliceutils"
	"github.com/nlpodyssey/cybertron/pkg/vocabulary"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

const (
	defaultMaxAnswerLength = 20
	defaultMinConfidence   = 0.1
	defaultMaxCandidates   = 3.0
	defaultMaxAnswers      = 3
)

// QuestionAnswering is a QuestionAnswering model.
type QuestionAnswering struct {
	// Model is the model used to answer questions.
	Model *bert.ModelForQuestionAnswering
	// Tokenizer is the tokenizer used to tokenize questions and passages.
	Tokenizer *wordpiecetokenizer.WordPieceTokenizer
}

// LoadQuestionAnswering returns a QuestionAnswering loading the model, the embeddings and the tokenizer from a directory.
func LoadQuestionAnswering(modelPath string) (*QuestionAnswering, error) {
	vocab, err := vocabulary.NewFromFile(filepath.Join(modelPath, "vocab.txt"))
	if err != nil {
		return nil, fmt.Errorf("failed to load vocabulary for question-answering: %w", err)
	}
	tokenizer := wordpiecetokenizer.New(vocab)

	m, err := nn.LoadFromFile[*bert.ModelForQuestionAnswering](path.Join(modelPath, "spago_model.bin"))
	if err != nil {
		return nil, fmt.Errorf("failed to load bart model: %w", err)
	}

	return &QuestionAnswering{
		Model:     m,
		Tokenizer: tokenizer,
	}, nil
}

// Answer returns the answers for the given question and passage.
// The options may assume default values if those are not set.
func (qa *QuestionAnswering) Answer(_ context.Context, question string, passage string, opts *questionanswering.Options) (questionanswering.Response, error) {
	checkOptions(opts)

	qt, pt := qa.tokenize(question, passage)
	if l, k := len(qt)+len(pt), qa.Model.Bert.Config.MaxPositionEmbeddings; l > k {
		return questionanswering.Response{}, fmt.Errorf("%w: %d > %d", questionanswering.ErrInputSequenceTooLong, l, k)
	}

	starts, ends := qa.Model.Answer(concat(qt, pt))
	starts, ends = adjustLogitsForInference(starts, ends, qt, pt)
	startsIdx := getBestIndices(extractScores(starts), opts.MaxCandidates)
	endsIdx := getBestIndices(extractScores(ends), opts.MaxCandidates)
	candidates := searchCandidates(startsIdx, endsIdx, starts, ends, pt, passage, opts.MaxAnswerLength)
	answers := filterUnlikelyCandidates(candidates, opts.MinScore)

	if len(answers) == 0 {
		return questionanswering.Response{}, nil
	}

	sort.Slice(answers, func(i, j int) bool {
		return answers[i].Score > answers[j].Score
	})

	if len(answers) > opts.MaxAnswers {
		answers = answers[:opts.MaxAnswers]
	}

	return questionanswering.Response{
		Answers: answers,
	}, nil
}

func checkOptions(opts *questionanswering.Options) {
	if opts.MaxAnswers == 0 {
		opts.MaxAnswers = defaultMaxAnswers
	}
	if opts.MaxAnswerLength == 0 {
		opts.MaxAnswerLength = defaultMaxAnswerLength
	}
	if opts.MaxCandidates == 0 {
		opts.MaxCandidates = defaultMaxCandidates
	}
	if opts.MinScore == 0 {
		opts.MinScore = defaultMinConfidence
	}
}

// tokenize splits the question and passage into tokens.
func (qa *QuestionAnswering) tokenize(question string, passage string) (qt []tokenizers.StringOffsetsPair, pt []tokenizers.StringOffsetsPair) {
	qt = qa.Tokenizer.Tokenize(question)
	pt = qa.Tokenizer.Tokenize(passage)
	return
}

// concat concatenates the question and passage tokens.
func concat(question, passage []tokenizers.StringOffsetsPair) []string {
	cls := wordpiecetokenizer.DefaultClassToken
	sep := wordpiecetokenizer.DefaultSequenceSeparator
	tokenized := append([]string{cls}, append(tokenizers.GetStrings(question), sep)...)
	tokenized = append(tokenized, append(tokenizers.GetStrings(passage), sep)...)
	return tokenized
}

// adjustLogitsForInference adjusts the logits for inference.
func adjustLogitsForInference(starts, ends []mat.Tensor, question, passage []tokenizers.StringOffsetsPair) ([]mat.Tensor, []mat.Tensor) {
	passageStartIndex := len(question) + 2 // the offset is for [CLS] and [SEP] tokens
	passageEndIndex := passageStartIndex + len(passage)
	return starts[passageStartIndex:passageEndIndex], ends[passageStartIndex:passageEndIndex]
}

// extractScores extracts the scores from the logits.
func extractScores(logits []mat.Tensor) []float64 {
	scores := make([]float64, len(logits))
	for i, node := range logits {
		scores[i] = node.Value().Item().F64()
	}
	return scores
}

// getBestIndices returns the best indices from the given scores.
func getBestIndices(logits []float64, size int) []int {
	s := sliceutils.NewIndexedSlice(logits)
	sort.Sort(sort.Reverse(s))
	if len(s.Indices) < size {
		return s.Indices
	}
	return s.Indices[:size]
}

// searchCandidates searches the candidates from the given starts and ends logits.
func searchCandidates(startsIdx, endsIdx []int, starts, ends []mat.Tensor, pt []tokenizers.StringOffsetsPair, passage string, maxLen int) []questionanswering.Answer {
	candidates := make([]questionanswering.Answer, 0)
	scores := make([]float64, 0) // the scores are aligned with the candidate answers
	for _, startIndex := range startsIdx {
		for _, endIndex := range endsIdx {
			switch {
			case endIndex < startIndex:
				continue
			case endIndex-startIndex+1 > maxLen:
				continue
			default:
				startOffset := pt[startIndex].Offsets.Start
				endOffset := pt[endIndex].Offsets.End
				scores = append(scores, ag.Add(starts[startIndex], ends[endIndex]).Value().Item().F64())
				candidates = append(candidates, questionanswering.Answer{
					Text:  strings.Trim(string([]rune(passage)[startOffset:endOffset]), " "),
					Start: startOffset,
					End:   endOffset,
				})
			}
		}
	}
	for i, prob := range mat.NewDense[float64](mat.WithBacking(scores)).Softmax().Data().F64() {
		candidates[i].Score = prob
	}
	return candidates
}

// filterUnlikelyCandidates filters the candidates that are unlikely to be the answer.
func filterUnlikelyCandidates(candidates []questionanswering.Answer, min float64) []questionanswering.Answer {
	answers := make([]questionanswering.Answer, 0)
	for _, candidate := range candidates {
		if candidate.Score >= min {
			answers = append(answers, candidate)
		}
	}
	return answers
}
