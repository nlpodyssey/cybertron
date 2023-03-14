// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package languagemodeling

import (
	"context"
	"errors"
)

const (
	// DefaultModel is a BERT pretrained model on English language using a masked language modeling (MLM) objective.
	// Model card: https://huggingface.co/bert-base-cased
	DefaultModel = "bert-base-cased"

	// DefaultItalianModel is a BERT pretrained model on Italian language using a masked language modeling (MLM) objective.
	// Model card: https://huggingface.co/dbmdz/bert-base-italian-cased
	DefaultItalianModel = "dbmdz/bert-base-italian-cased"
)

// ErrInputSequenceTooLong means that pre-processing the input text
// produced a sequence that exceeds the maximum allowed length.
var ErrInputSequenceTooLong = errors.New("input sequence too long")

// Interface defines the main functions for language modelling.
type Interface interface {
	// Predict returns the prediction of the given example.
	Predict(ctx context.Context, text string, parameters Parameters) (Response, error)
}

// Parameters contains the parameters for language modeling.
type Parameters struct {
	// K is the number of returned predictions per token
	K int
}

// Token is a labeled text token.
type Token struct {
	Start  int
	End    int
	Words  []string
	Scores []float64
}

// Response contains the response from language modelling..
type Response struct {
	Tokens []Token
}
