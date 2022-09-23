// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package textclassification

import (
	"context"
)

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

// Filter returns a function to filter the classification response with respect to two parameters, keepThreshold and
// keepSumThreshold, which are used to check whether to consider the single prediction, and to check whether the sum
// of all collected prediction scores allows a result to be returned or not, respectively.
func Filter(keepThreshold, keepSumThreshold float64) func(r Response) Response {
	return func(response Response) Response {
		n, sum := -1, 0.0
		for i, score := range response.Scores {
			if score >= keepThreshold {
				n = i
				sum += score
			}
		}
		if n == -1 || sum < keepSumThreshold {
			return Response{}
		}
		return Response{
			Labels: response.Labels[:n+1],
			Scores: response.Scores[:n+1],
		}
	}
}
