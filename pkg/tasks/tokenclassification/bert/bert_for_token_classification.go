// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/spago/ag"
)

type ModelForTokenClassification struct {
	*bert.ModelForTokenClassification
}

// Classify returns the logits for each token.
func (m *ModelForTokenClassification) Classify(tokens []string) []ag.Node {
	return m.Classifier.Forward(m.EncodeAndReduce(tokens)...)
}

func (m *ModelForTokenClassification) EncodeAndReduce(tokens []string) []ag.Node {
	encoded := m.Bert.EncodeTokens(tokens)

	result := make([]ag.Node, 0, len(tokens))
	for i, token := range tokens {
		if isSpecialToken(token) {
			encoded[i].Value() // important
			continue
		}
		result = append(result, encoded[i])
	}
	return result
}

func isSpecialToken(token string) bool {
	return strings.HasPrefix(token, wordpiecetokenizer.DefaultSplitPrefix) ||
		strings.EqualFold(token, wordpiecetokenizer.DefaultClassToken) ||
		strings.EqualFold(token, wordpiecetokenizer.DefaultSequenceSeparator) ||
		strings.EqualFold(token, wordpiecetokenizer.DefaultMaskToken)
}
