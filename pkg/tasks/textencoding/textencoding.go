// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package textencoding

import (
	"context"

	"github.com/nlpodyssey/spago/mat"
)

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
