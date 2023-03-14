// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package zeroshotclassifier

import (
	"context"
	"errors"
)

const (
	// DefaultModel is a model for Natural Language Inference (NLI) that can be used for zero-shot classification.
	// Model card: https://huggingface.co/valhalla/distilbart-mnli-12-3
	DefaultModel = "valhalla/distilbart-mnli-12-3"
)

const (
	// DefaultHypothesisTemplate is the string template that is interpolated with each class to predict.
	DefaultHypothesisTemplate = "This example is {}."
)

// ErrInputSequenceTooLong means that pre-processing the input text
// produced a sequence that exceeds the maximum allowed length.
var ErrInputSequenceTooLong = errors.New("input sequence too long")

// Interface defines the main functions for zero-shot classification task.
type Interface interface {
	// Classify returns the classification of the given example.
	Classify(ctx context.Context, text string, parameters Parameters) (Response, error)
}

// Parameters contains the parameters for zero-shot classification.
type Parameters struct {
	// A list of strings that are potential classes for inputs. (required)
	CandidateLabels []string
	// HypothesisTemplate is the string template that is interpolated with each class to predict.
	// For example, “this text is about {}”. (optional)
	HypothesisTemplate string
	// MultiLabel set to True if classes can overlap (default: false)
	MultiLabel bool
}

// Response contains the response from zero-shot classification.
type Response struct {
	// The list of labels sent in the request, sorted in descending order
	// by probability that the input corresponds to the label.
	Labels []string
	// a list of floats that correspond the probability of label, in the same order as labels.
	Scores []float64
}
