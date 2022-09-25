// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional copyright notes in the package README.

package generationutils

import (
	"encoding/binary"
	"math"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

var floatNegInf = float.Interface(math.Inf(-1))

func (b *BeamSearchDecoder) adjustPrediction(inputIDs [][]int, scores []mat.Matrix) []mat.Matrix {
	if b.Config.MinLength >= 0 && b.Config.EOSTokenID >= 0 {
		scores = b.processMinLengthScores(inputIDs, scores)
	}
	if len(b.Config.BadWordsIDs) > 0 {
		scores = b.processBadWordsScores(inputIDs, scores)
	}
	if b.Config.NoRepeatNGramSize > 0 {
		scores = b.processNoRepeatNGramScores(inputIDs, scores)
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
	for idx, batchBannedTokens := range bannedTokens {
		for _, tokenID := range batchBannedTokens {
			scores[idx].SetVecScalar(tokenID, floatNegInf)
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
	for _, n := range scores {
		n.SetVecScalar(eosTokenID, floatNegInf)
	}

	return scores
}

func (b *BeamSearchDecoder) processNoRepeatNGramScores(inputIDs [][]int, scores []mat.Matrix) []mat.Matrix {
	numBatchHypotheses := len(scores)
	curLen := len(inputIDs[0])
	bannedBatchTokens := b.calcBannedNGramTokens(inputIDs, numBatchHypotheses, curLen)
	if len(bannedBatchTokens) == 0 {
		return scores
	}

	for i, bannedTokens := range bannedBatchTokens {
		sc := scores[i]
		for _, j := range bannedTokens {
			sc.SetVecScalar(j, floatNegInf)
		}
	}
	return scores
}

func (b *BeamSearchDecoder) calcBannedNGramTokens(prevInputIDs [][]int, numHypos, curLen int) [][]int {
	nGramSize := b.Config.NoRepeatNGramSize
	if curLen+1 < nGramSize {
		return nil // No banned tokens
	}

	generatedNGrams := b.getNGrams(prevInputIDs, numHypos)

	bannedTokens := make([][]int, numHypos)
	for i := range bannedTokens {
		bannedTokens[i] = b.getGeneratedNGrams(generatedNGrams[i], prevInputIDs[i], curLen)
	}

	return bannedTokens
}

func (b *BeamSearchDecoder) getGeneratedNGrams(bannedNGrams map[string][]int, prevInputIDs []int, curLen int) []int {
	nGramSize := b.Config.NoRepeatNGramSize
	startIndex := curLen + 1 - nGramSize
	nGram := prevInputIDs[startIndex:curLen]
	key := intSliceMapKey(nGram)
	return bannedNGrams[key]
}

func (b *BeamSearchDecoder) getNGrams(prevInputIDs [][]int, numHypos int) []map[string][]int {
	nGramSize := b.Config.NoRepeatNGramSize
	generatedNGrams := make([]map[string][]int, numHypos)
	for idx := range generatedNGrams {
		genTokens := prevInputIDs[idx]

		mapSize := len(genTokens) - nGramSize + 1
		if mapSize == 0 {
			continue
		}

		generatedNGrams[idx] = make(map[string][]int, mapSize)
		generatedNGram := generatedNGrams[idx]

		for nGramStart := 0; nGramStart < mapSize; nGramStart++ {
			nGram := genTokens[nGramStart : nGramStart+nGramSize]
			prevNGram := nGram[:len(nGram)-1]
			key := intSliceMapKey(prevNGram)
			generatedNGram[key] = append(generatedNGram[key], nGram[len(nGram)-1])
		}
	}
	return generatedNGrams
}

func intSliceMapKey(v []int) string {
	bs := make([]byte, len(v)*8) // binary representation of len(s) uint64
	for i, x := range v {
		binary.LittleEndian.PutUint64(bs[8*i:], uint64(x))
	}
	return string(bs)
}

// intSliceEqual returns whether the two slices are equal, or not.
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
