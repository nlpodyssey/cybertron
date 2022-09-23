// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package textclassification

import "context"

const (
	// DefaultModelForItalianNewsClassification is a model fine-tuned for news headlines classification in Italian.
	// It predicts the top-level category of the IPTC subject taxonomy: https://cv.iptc.org/newscodes/subjectcode
	// Model card: https://huggingface.co/nlpodyssey/bert-italian-uncased-iptc-headlines
	DefaultModelForItalianNewsClassification = "nlpodyssey/bert-italian-uncased-iptc-headlines"
)

// Interface defines the main functions for text classification task.
type Interface interface {
	// Classify returns the classification of the given example.
	Classify(ctx context.Context, text string) (Response, error)
}

// Response contains the response from text classification.
type Response struct {
	// The list of labels sent in the request, sorted in descending order
	// by probability that the input corresponds to the label.
	Labels []string
	// a list of floats that correspond the probability of label, in the same order as labels.
	Scores []float64
}
