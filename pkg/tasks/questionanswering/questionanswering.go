// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package questionanswering

import (
	"context"
	"errors"
)

const (
	// DefaultEnglishModel is a model for extractive question-answering for the English language.
	// Model card: https://huggingface.co/deepset/bert-base-cased-squad2
	DefaultEnglishModel = "deepset/bert-base-cased-squad2"

	// DefaultItalianModel is a model for extractive question-answering for the Italian language.
	// Model card: https://huggingface.co/mrm8488/bert-italian-finedtuned-squadv1-it-alfa
	DefaultItalianModel = "mrm8488/bert-italian-finedtuned-squadv1-it-alfa"
)

// ErrInputSequenceTooLong means that pre-processing the input text
// produced a sequence that exceeds the maximum allowed length.
var ErrInputSequenceTooLong = errors.New("input sequence too long")

// Interface defines the main functions for question-answering task.
type Interface interface {
	Answer(ctx context.Context, question string, passage string, opts *Options) (Response, error)
}

// Options defines the options for question-answering task.
type Options struct {
	// MaxAnswers is the maximum number of answers to return.
	MaxAnswers int
	// MaxAnswerLength is the maximum length of answers to return.
	MaxAnswerLength int
	// MinScore is the minimum confidence score for answers to return.
	MinScore float64
	// MaxCandidates
	MaxCandidates int
}

// Answer represents the single answer of a question.
type Answer struct {
	// Text is the span of text containing the answer.
	Text string
	// Start is the start index of the answer in the passage.
	Start int
	// End is the end index of the answer in the passage.
	End int
	// Score is the score of the answer.
	Score float64
}

// Response contains the response from question-answering task.
type Response struct {
	// Answers contains the list of answers.
	Answers []Answer
}
