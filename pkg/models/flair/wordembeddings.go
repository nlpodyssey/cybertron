// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/embeddings"
)

type WordEmbeddings struct {
	*embeddings.Model[string]
}

var _ TokensEncoder = &WordEmbeddings{}

func init() {
	gob.Register(&WordEmbeddings{})
}
