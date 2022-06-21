// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

// Tokenize returns the token IDs of the input text applying the EOS pad token.
func (m *Text2Text) Tokenize(text string) []int {
	return append(m.Tokenizer.TokensToIDs(m.Tokenizer.Tokenize(text)), m.Model.Bart.Config.EosTokenID)
}

// Detokenize returns the text of the input token IDs removing the padding token.
func (m *Text2Text) Detokenize(tokenIds []int) string {
	stripBadTokens := func(tokenIds []int) []int {
		config := m.Model.Bart.Config
		result := make([]int, 0, len(tokenIds))
		for _, id := range tokenIds {
			if id == config.EosTokenID || id == config.PadTokenID || id == config.BosTokenID ||
				id == config.DecoderStartTokenID {
				continue
			}
			result = append(result, id)
		}
		return result
	}

	return m.Tokenizer.Detokenize(m.Tokenizer.IDsToTokens(stripBadTokens(tokenIds)))
}
