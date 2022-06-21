// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

// tokenize returns the tokenized text and applies start and end padding.
func (m *ZeroShotClassifier) tokenize(text string, startTokenID, endTokenID int) ([]int, error) {
	encoded, err := m.Tokenizer.Encode(text)
	if err != nil {
		return nil, err
	}

	tokenized := make([]int, len(encoded.IDs)+2)
	tokenized[0] = startTokenID
	copy(tokenized[1:len(tokenized)-1], encoded.IDs)
	tokenized[len(tokenized)-1] = endTokenID

	return tokenized, nil
}
