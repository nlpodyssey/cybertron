// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package textclassification

// Interface defines the main functions for text classification task.
type Interface interface {
	// Classify returns the classification of the given example.
	Classify(text string) (Response, error)
}

// Response contains the response from text classification.
type Response struct {
	// The list of labels sent in the request, sorted in descending order
	// by probability that the input corresponds to the label.
	Labels []int // string
	// a list of floats that correspond the probability of label, in the same order as labels.
	Scores []float64
}
