// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import "github.com/nlpodyssey/cybertron/pkg/tokenizers/sentencepiece"

type SentencePieceTokenizer struct {
	*sentencepiece.Tokenizer
	EosTokenID, BosTokenID, PadTokenID, DecoderStartTokenID int
}

// Tokenize returns the token IDs of the input text applying the EOS pad token.
func (m *SentencePieceTokenizer) Tokenize(text string) ([]int, error) {
	return append(m.Tokenizer.TokensToIDs(m.Tokenizer.Tokenize(text)), m.EosTokenID), nil
}

// Detokenize returns the text of the input token IDs removing the padding token.
func (m *SentencePieceTokenizer) Detokenize(tokenIds []int) string {
	stripBadTokens := func(tokenIds []int) []int {
		result := make([]int, 0, len(tokenIds))
		for _, id := range tokenIds {
			if id == m.EosTokenID || id == m.PadTokenID || id == m.BosTokenID || id == m.DecoderStartTokenID {
				continue
			}
			result = append(result, id)
		}
		return result
	}

	return m.Tokenizer.Detokenize(m.Tokenizer.IDsToTokens(stripBadTokens(tokenIds)))
}
