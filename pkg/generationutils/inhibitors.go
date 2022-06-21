// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional copyright notes in the package README.

package generationutils

import (
	"math"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

func (b *BeamSearchDecoder) adjustPrediction(inputIDs [][]int, scores []mat.Matrix) []mat.Matrix {
	if b.Config.MinLength >= 0 && b.Config.EOSTokenID >= 0 {
		scores = b.processMinLengthScores(inputIDs, scores)
	}
	if len(b.Config.BadWordsIDs) > 0 {
		scores = b.processBadWordsScores(inputIDs, scores)
	}
	return scores
}

func (b *BeamSearchDecoder) processBadWordsScores(inputIDs [][]int, scores []mat.Matrix) []mat.Matrix {
	BadWordsIDs := make([][]int, 0, len(b.Config.BadWordsIDs))
	for _, v := range b.Config.BadWordsIDs {
		if len(v) == 1 && v[0] == b.Config.EOSTokenID {
			continue
		}
		BadWordsIDs = append(BadWordsIDs, v)
	}

	// Calculate banned bad words IDs
	bannedTokens := make([][]int, 0, len(inputIDs))
	for _, slice := range inputIDs {
		bannedTokensSlice := make([]int, 0, len(slice))
		for _, bannedTokenSeq := range BadWordsIDs {
			if !bannedTokensMatch(slice, bannedTokenSeq[:len(bannedTokenSeq)-1]) {
				continue
			}
			bannedTokensSlice = append(bannedTokensSlice, bannedTokenSeq[len(bannedTokenSeq)-1])
		}
		bannedTokens = append(bannedTokens, bannedTokensSlice)
	}

	// Set scores to -Inf for banned tokens
	negInf := float.Interface(math.Inf(-1))
	for idx, batchBannedTokens := range bannedTokens {
		for _, tokenID := range batchBannedTokens {
			scores[idx].SetVecScalar(tokenID, negInf)
		}
	}

	return scores
}

func bannedTokensMatch(prevTokens []int, bannedTokens []int) bool {
	if len(bannedTokens) == 0 {
		// If bad word tokens is just one token always ban it.
		return true
	}
	if len(bannedTokens) > len(prevTokens) {
		// If bad word tokens are longer then prev input_ids they can't be equal.
		return false
	}
	return intSliceEqual(prevTokens[len(prevTokens)-len(bannedTokens):], bannedTokens)
}

func (b *BeamSearchDecoder) processMinLengthScores(inputIDs [][]int, scores []mat.Matrix) []mat.Matrix {
	curLen := len(inputIDs[0])
	if curLen >= b.Config.MinLength {
		return scores
	}

	eosTokenID := b.Config.EOSTokenID
	negInf := float.Interface(math.Inf(-1))
	for _, n := range scores {
		n.SetVecScalar(eosTokenID, negInf)
	}

	return scores
}

// intSliceEqual returns whether the two sliceutils are equal, or not.
func intSliceEqual(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i, va := range a {
		if va != b[i] {
			return false
		}
	}
	return true
}
