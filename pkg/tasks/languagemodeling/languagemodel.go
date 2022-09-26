// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package languagemodeling

import "context"

const (
	// DefaultModel is a BERT pretrained model on English language using a masked language modeling (MLM) objective.
	// Model card: https://huggingface.co/bert-base-cased
	DefaultModel = "bert-base-cased"

	// DefaultItalianModel is a BERT pretrained model on Italian language using a masked language modeling (MLM) objective.
	// Model card: https://huggingface.co/dbmdz/bert-base-italian-cased
	DefaultItalianModel = "dbmdz/bert-base-italian-cased"
)

// Interface defines the main functions for language modelling.
type Interface interface {
	// Predict returns the prediction of the given example.
	Predict(ctx context.Context, text string) (Response, error)
}

// Response contains the response from language modelling.
type Response struct {
	Words  []string
	Scores []float64
}
