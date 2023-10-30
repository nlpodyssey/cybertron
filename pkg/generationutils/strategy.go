// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package generationutils

import (
	"sort"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/mat/rand"
)

// DecodingStrategyFunc returns the next tokens to be generated.
type DecodingStrategyFunc func(tokensScores []mat.Matrix, resultSize int) []*ScoredToken

// SelectNextTopK returns the next tokens to be generated.
func SelectNextTopK(tokensScores []mat.Matrix, resultSize int) []*ScoredToken {
	if len(tokensScores) == 0 || resultSize <= 0 {
		return nil
	}
	switch tokensScores[0].Data().BitSize() {
	case 32:
		return selectNextTopK[float32](tokensScores, resultSize)
	case 64:
		return selectNextTopK[float64](tokensScores, resultSize)
	default:
		panic("generationutils: unexpected matrix data bit size")
	}
}

// selectNextTopK returns the next tokens to be generated.
func selectNextTopK[T float.DType](tokensScores []mat.Matrix, resultSize int) []*ScoredToken {
	arena := make([]ScoredToken, resultSize)
	result := make([]*ScoredToken, 0, resultSize)

	var minScore float64
	minIndex := -1

	for beamIndex, n := range tokensScores {
		for tokenIndex, tScore := range mat.Data[T](n) {
			score := float64(tScore)

			if len(result) < resultSize {
				if minIndex == -1 || score < minScore {
					minScore = score
					minIndex = len(result)
				}

				st := &arena[0]
				arena = arena[1:]

				st.BeamIndex = beamIndex
				st.TokenIndex = tokenIndex
				st.Score = score

				result = append(result, st)
				continue
			}

			if score <= minScore {
				continue
			}

			// Replace the scored token with minimum score with the new one
			st := result[minIndex]
			st.BeamIndex = beamIndex
			st.TokenIndex = tokenIndex
			st.Score = score

			// Find the new minimum
			minScore = result[0].Score
			minIndex = 0
			for i, v := range result {
				if v.Score < minScore {
					minScore = v.Score
					minIndex = i
				}
			}
		}
	}

	sort.SliceStable(result, func(i, j int) bool {
		return result[i].Score > result[j].Score
	})

	return result
}

// SelectNextMultinomial returns the next tokens to be generated.
func SelectNextMultinomial(tokensScores []mat.Matrix, resultSize int) []*ScoredToken {
	result := make([]*ScoredToken, 0, resultSize*len(tokensScores))

	for beamIndex, m := range tokensScores {
		nextIndices := multinomialSample(m.Softmax(), resultSize)
		for _, nextIndex := range nextIndices {
			result = append(result, &ScoredToken{
				BeamIndex:  beamIndex,
				TokenIndex: nextIndex,
				// FIXME: avoid casting to specific type
				Score: m.ScalarAt(nextIndex).F64(),
			})
		}
	}

	sort.SliceStable(result, func(i, j int) bool {
		return result[i].Score > result[j].Score
	})

	return result
}

// sample extracts the next index from the probability multinomial distribution.
func multinomialSample(probs mat.Matrix, numSamples int) []int {
	if numSamples > probs.Size() {
		panic("generationutils: cannot sample numSamples > probs.Size() samples")
	}
	// FIXME: avoid casting to specific type
	probsData := probs.Data().F64()
	samples := make([]int, 0, numSamples)
	samplesMap := make(map[int]struct{}, numSamples)

	for len(samples) < numSamples {
		p := rand.Float[float64]()

		for probIndex, prob := range probsData {
			p -= prob
			if p < 0 {
				if _, alreadySampled := samplesMap[probIndex]; !alreadySampled {
					samplesMap[probIndex] = struct{}{}
					samples = append(samples, probIndex)
				}
				break
			}
		}
	}

	return samples
}
