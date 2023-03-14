// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package textencoding

import (
	"context"
	"errors"

	"github.com/nlpodyssey/spago/mat"
)

const (
	// DefaultModel is a sentence-transformers model:
	// It maps sentences & paragraphs to dense vector space and can be used for tasks like clustering or semantic search.
	// Model card: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
	DefaultModel = "sentence-transformers/all-MiniLM-L6-v2"

	// DefaultModelMulti it's like the model before but multilingual:
	// it can be used to map 109 languages to a shared vector space.
	// It works well for finding translation pairs in multiple languages.
	// It doesn't perform so well for assessing the similarity of sentence pairs that are not translations of each other.
	// Model card: https://huggingface.co/sentence-transformers/LaBSE
	DefaultModelMulti = "sentence-transformers/LaBSE"
)

// ErrInputSequenceTooLong means that pre-processing the input text
// produced a sequence that exceeds the maximum allowed length.
var ErrInputSequenceTooLong = errors.New("input sequence too long")

// Interface defines the main functions for text encoding task.
type Interface interface {
	// Encode returns the encoded representation of the given example.
	Encode(ctx context.Context, text string, poolingStrategy int) (Response, error)
}

// Response contains the response from text classification.
type Response struct {
	// the encoded representation
	Vector mat.Matrix
}
