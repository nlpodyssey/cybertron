// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional copyright notes in the package README.

package generationutils

import (
	"math"
	"sort"
)

// hypothesis represents a single generated hypothesis, which is a sequence of
// Token IDs paired with a score.
type hypothesis struct {
	sequence []int
	score    float64
}

// hypotheses represents a list of hypotheses.
type hypotheses struct {
	config hypothesesConfig
	items  []*hypothesis
}

type hypothesesConfig struct {
	eosTokenID    int
	maxLength     int
	maxHypotheses int
	earlyStopping bool
	lengthPenalty float64
}

func newHypotheses(c Config) *hypotheses {
	return &hypotheses{
		config: hypothesesConfig{
			eosTokenID:    c.EOSTokenID,
			maxLength:     c.MaxLength,
			maxHypotheses: c.NumBeams,
			earlyStopping: c.EarlyStopping,
			lengthPenalty: c.LengthPenalty,
		},
		items: make([]*hypothesis, 0, c.NumBeams),
	}
}

// insert adds a new hypothesis to the list.
func (h *hypotheses) insert(item *hypothesis) {
	if len(h.items) == h.config.maxHypotheses && item.score <= h.items[len(h.items)-1].score {
		// the new hypothesis is worse than the worst one in the heap
		return
	}
	if len(h.items) == h.config.maxHypotheses {
		h.items = h.items[:len(h.items)-1]
	}

	buf := make([]int, len(item.sequence))
	copy(buf, item.sequence)
	h.items = append(h.items, &hypothesis{sequence: buf, score: item.score})

	sort.SliceStable(h.items, func(i, j int) bool {
		return h.items[i].score > h.items[j].score
	})
}

// isDone reports whether there are enough Hypotheses and none of the Hypotheses
// being generated can become better than the worst one in the heap.
func (h *hypotheses) isDone(bestSumLogProbs float64, curLen int) bool {
	if len(h.items) < h.config.maxHypotheses {
		return false
	}
	if h.config.earlyStopping {
		return true
	}
	curScore := bestSumLogProbs / math.Pow(float64(curLen), h.config.lengthPenalty)
	worstScore := h.items[len(h.items)-1].score
	return worstScore >= curScore
}
