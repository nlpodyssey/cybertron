// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import "github.com/nlpodyssey/cybertron/pkg/tokenizers/bpetokenizer"

type BPETokenizer struct {
	*bpetokenizer.BPETokenizer
	EosTokenID           int
	BosTokenID           int
	PadTokenID           int
	DecoderStartTokenID  int
	ExtraSpecialTokenIDs map[int]string
}

// Tokenize returns the token IDs of the input text applying the EOS pad token.
func (m *BPETokenizer) Tokenize(text string) ([]int, error) {
	encoded, err := m.BPETokenizer.Encode(text)
	if err != nil {
		return nil, err
	}

	tokenized := make([]int, len(encoded.IDs)+2)
	tokenized[0] = m.BosTokenID
	copy(tokenized[1:len(tokenized)-1], encoded.IDs)
	tokenized[len(tokenized)-1] = m.EosTokenID

	return tokenized, nil
}

// Detokenize returns the text of the input token IDs removing the padding token.
func (m *BPETokenizer) Detokenize(tokenIds []int, stripPaddingTokens bool) string {
	if !stripPaddingTokens {
		return m.BPETokenizer.Detokenize(tokenIds)
	}

	stripPaddingTokensFn := func(tokenIds []int) []int {
		result := make([]int, 0, len(tokenIds))
		for _, id := range tokenIds {
			if id == m.EosTokenID || id == m.PadTokenID || id == m.BosTokenID || id == m.DecoderStartTokenID {
				continue
			}
			result = append(result, id)
		}
		return result
	}

	return m.BPETokenizer.Detokenize(stripPaddingTokensFn(tokenIds))
}
