// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package text2text

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/utils/nullable"
)

const (
	// DefaultModelTemplateForMachineTranslation is the template for machine translations models.
	// You can use it as the basis for constructing the default model name for the desired source
	// and target languages pair (iso-a2), or you can use the convenient function DefaultModelForMachineTranslation.
	// Model card: https://huggingface.co/Helsinki-NLP
	DefaultModelTemplateForMachineTranslation = "Helsinki-NLP/opus-mt-%s-%s"

	// DefaultModelForTextParaphrasing is a summarization model fine-tuned for text paraphrasing.
	// Model card: https://huggingface.co/tuner007/pegasus_paraphrase
	DefaultModelForTextParaphrasing = "tuner007/pegasus_paraphrase"

	// DefaultModelForTextSummarization is a summarization model.
	// Model card: https://huggingface.co/facebook/bart-large-cnn
	DefaultModelForTextSummarization = "facebook/bart-large-cnn"

	// DefaultModelForTextSummarization2 is a summarization model.
	// Model card: https://huggingface.co/google/pegasus-multi_news
	DefaultModelForTextSummarization2 = "google/pegasus-multi_news"

	// DefaultModelForExtremeTextSummarization is a summarization model that tries to generate one-sentence answering
	// the question “What is the article about?”.
	// Model card: https://huggingface.co/facebook/bart-large-xsum
	DefaultModelForExtremeTextSummarization = "facebook/bart-large-xsum"

	// DefaultModelForAbstractiveQuestionAnswering is a summarization model fine-tuned for answer generation.
	// Model card: https://huggingface.co/vblagoje/bart_lfqa/tree/main
	DefaultModelForAbstractiveQuestionAnswering = "vblagoje/bart_lfqa"

	// DefaultModelForKeywordsGeneration is a text generation model that produces a concatenated sequence of keyphrases.
	// Model card: https://huggingface.co/bloomberg/KeyBART
	DefaultModelForKeywordsGeneration = "bloomberg/KeyBART"
)

// DefaultModelForMachineTranslation specializes the model template for the source and target languages (iso-a2).
func DefaultModelForMachineTranslation(source, target string) string {
	return fmt.Sprintf(DefaultModelTemplateForMachineTranslation, source, target)
}

// Interface defines the main functions for the Text2Text task.
type Interface interface {
	// Generate generates text (e.g. translation, summarization, paraphrase) from the given input.
	Generate(ctx context.Context, text string, opts *Options) (Response, error)
}

// Options defines the options for generating text.
type Options struct {
	// Temperature is the temperature used for sampling.
	Temperature nullable.Type[float64]
	// Sample is whether to sample or greedy generation.
	Sample nullable.Type[bool]
	// TopK is the number of top-k candidates to be considered during generation.
	TopK nullable.Type[int]
	// TopP is the top-p candidates to be considered during generation.
	TopP nullable.Type[float64]
}

// Response contains the result of the text generation.
type Response struct {
	// Texts contains the generated texts.
	Texts []string
	// a list of floats that correspond the score of the generated text, in the same order as texts.
	Scores []float64
}

// ErrInputSequenceTooLong means that pre-processing the input text
// produced a sequence that exceeds the maximum allowed length.
var ErrInputSequenceTooLong = errors.New("input sequence too long")

// DefaultOptions returns the default options for generating text.
func DefaultOptions() *Options {
	return &Options{
		Temperature: nullable.Type[float64]{Value: 1.0, Valid: true},
		Sample:      nullable.Type[bool]{Value: false, Valid: true},
		TopK:        nullable.Type[int]{Valid: false},
		TopP:        nullable.Type[float64]{Valid: false},
	}
}

// DefaultOptionsForTextParaphrasing returns the default options for generating text with sampling.
func DefaultOptionsForTextParaphrasing() *Options {
	return &Options{
		Temperature: nullable.Type[float64]{Value: 1.5, Valid: true},
		Sample:      nullable.Type[bool]{Value: true, Valid: true},
		TopK:        nullable.Type[int]{Value: 120, Valid: true},
		TopP:        nullable.Type[float64]{Value: 0.98, Valid: false},
	}
}

// PrepareInputForAbstractiveQuestionAnswering returns text to be input to the DefaultModelForAbstractiveQuestionAnswering.
func PrepareInputForAbstractiveQuestionAnswering(question string, passages []string) string {
	return fmt.Sprintf("question: %s context: %s", question, "<P> "+strings.Join(passages, " <P> "))
}
