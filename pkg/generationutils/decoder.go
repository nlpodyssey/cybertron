// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional copyright notes in the package README.

// Package generationutils implements a decoding search algorithm for conditional generation.
package generationutils

import (
	"math"

	"context"
	"github.com/nlpodyssey/spago/mat"
	"github.com/rs/zerolog/log"
)

// BeamSearchDecoder is an implementations of a decoding search algorithm for conditional generation.
type BeamSearchDecoder struct {
	// Config is the configuration of the beam decoder.
	Config Config
	// PredictNext is a function that predicts the next tokens given the current tokens.
	PredictNext PredictNextFunc
	// SelectNext is a function that selects the next tokens given the current tokens.
	SelectNext DecodingStrategyFunc
}

// PredictNextFunc is a function that predicts the next token scores for a given input.
type PredictNextFunc func(decodingInputIDs [][]int, lastBeamIndices []int) []mat.Matrix

// ScoredToken associates a score to a token identified by its
// (beam-index, token-index) position.
type ScoredToken struct {
	BeamIndex  int
	TokenIndex int
	Score      float64
}

// Decode generates sequences for model with a language modeling head, using
// beam-search decoding.
func (b *BeamSearchDecoder) Decode(ctx context.Context) ([][]int, []float64) {
	var (
		hs          = newHypotheses(b.Config)
		beamIndices = make([]int, 1, b.Config.NumBeams)
		sumLogProbs = make([]float64, 1, b.Config.NumBeams)
		inputIDs    = make([][]int, 1, b.Config.NumBeams)
		isDone      = false
	)

	inputIDs[0] = []int{b.Config.DecoderStartTokenID}

Loop:
	for curLen := 1; curLen < b.Config.MaxLength; curLen++ {
		candidates := b.generateCandidates(inputIDs, beamIndices, sumLogProbs)
		selected := b.SelectNext(candidates, b.Config.NumBeams*2)
		inputIDs, beamIndices, sumLogProbs = b.process(inputIDs, selected, func(sequence []int, sumLogProb float64) {
			// add to hypothesis if end of sentence
			hs.insert(&hypothesis{
				sequence: sequence,
				score:    sumLogProb / math.Pow(float64(len(sequence)), b.Config.LengthPenalty),
			})
		})
		if isDone = hs.isDone(selected[0].Score, curLen); isDone {
			break
		}

		select {
		case <-ctx.Done():
			log.Trace().Msg("context done, returning what has been computed so far.")
			break Loop
		default:
		}
	}

	if !isDone {
		// add remaining hypotheses
		for beamID := 0; beamID < b.Config.NumBeams; beamID++ {
			sequence := inputIDs[beamID]
			hs.insert(&hypothesis{
				sequence: sequence,
				score:    sumLogProbs[beamID] / math.Pow(float64(len(sequence)), b.Config.LengthPenalty),
			})
		}
	}

	return hs.prepareOutput()
}

func (b *BeamSearchDecoder) generateCandidates(inputIDs [][]int, beamIndices []int, beamScores []float64) []mat.Matrix {
	tokensScores := b.PredictNext(inputIDs, beamIndices)
	tokensScores = b.adjustPrediction(inputIDs, tokensScores)

	// Add beam scores to the token scores.
	_ = tokensScores[len(beamScores)-1]
	for i, beamScore := range beamScores {
		tokensScores[i].AddScalarInPlace(beamScore)
	}
	return tokensScores
}
