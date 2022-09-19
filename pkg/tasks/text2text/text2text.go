// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package text2text

import (
	"context"

	"github.com/nlpodyssey/cybertron/pkg/utils/nullable"
)

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
