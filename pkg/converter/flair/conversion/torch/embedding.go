// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import "github.com/nlpodyssey/spago/mat"

type Embedding struct {
	NumEmbeddings int
	EmbeddingDim  int
	Weight        []mat.Matrix
	NormType      int
}

func EmbeddingFromPretrained(embeddings []mat.Matrix, dim int) *Embedding {
	return &Embedding{
		NumEmbeddings: len(embeddings),
		EmbeddingDim:  dim,
		Weight:        embeddings,
		NormType:      2,
	}
}
