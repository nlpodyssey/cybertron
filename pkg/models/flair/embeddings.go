// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/linear"
)

type Embeddings struct {
	nn.Module
	TokensEncoder []TokensEncoder
	Projection    *linear.Model
}

type TokensEncoder interface {
	Encode(keys []string) []ag.Node
}

func (m *Embeddings) Encode(tokens []string) []ag.Node {
	encoded := make([][]ag.Node, len(tokens))
	for _, encoder := range m.TokensEncoder {
		for i, encoding := range encoder.Encode(tokens) {
			encoded[i] = append(encoded[i], encoding)
		}
	}

	buf := make([]ag.Node, len(tokens))
	for i, encoding := range encoded {
		if len(encoding) == 1 {
			buf[i] = encoding[0]
			continue
		}
		buf[i] = ag.Concat(encoding...)
	}
	return m.Projection.Forward(buf...)
}
