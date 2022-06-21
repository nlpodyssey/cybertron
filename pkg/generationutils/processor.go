// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional copyright notes in the package README.

package generationutils

// process elaborates the selected tokens and returns the next input IDs, beam indices and sum log probabilities.>>
func (b *BeamSearchDecoder) process(inputIDs [][]int, scoredTokens []*ScoredToken, onEndOfSentence func(sequence []int, sumLogProb float64)) (
	newInputIDs [][]int,
	newBeamIndices []int,
	newSumLogProbs []float64,
) {
	numBeams := b.Config.NumBeams
	eosTokenID := b.Config.EOSTokenID

	newBeamIndices = make([]int, numBeams, b.Config.NumBeams)
	newSumLogProbs = make([]float64, numBeams, b.Config.NumBeams)
	newBeamTokens := make([]int, numBeams, b.Config.NumBeams)

	// next tokens for this sentence
	beamIdx := 0

	for beamTokenRank, scoredToken := range scoredTokens {
		// add to generated Hypotheses if end of sentence
		if eosTokenID >= 0 && scoredToken.TokenIndex == eosTokenID {
			// if the token does not belong to top numBeams tokens, it should not be added
			if beamTokenRank >= numBeams {
				continue
			}
			onEndOfSentence(inputIDs[scoredToken.BeamIndex], scoredToken.Score)
		} else {
			// add next predicted token since it is not eos_token
			newSumLogProbs[beamIdx] = scoredToken.Score
			newBeamTokens[beamIdx] = scoredToken.TokenIndex
			newBeamIndices[beamIdx] = scoredToken.BeamIndex
			beamIdx++
		}

		// once the generation for next step is full, don't add more tokens to it.
		if beamIdx == numBeams {
			break
		}
	}

	// prepares the inputs for the next decoding step
	newInputIDs = make([][]int, len(newBeamIndices))
	for i, beamIndex := range newBeamIndices {
		prevValue := inputIDs[beamIndex]
		newInputIDs[i] = make([]int, 0, len(prevValue)+1)
		newInputIDs[i] = append(newInputIDs[i], prevValue...)
		newInputIDs[i] = append(newInputIDs[i], newBeamTokens[i])
	}
	return
}
