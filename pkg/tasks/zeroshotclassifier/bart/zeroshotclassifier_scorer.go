// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"github.com/nlpodyssey/spago/mat"
)

func (m *ZeroShotClassifier) score(premise []int, multiClass bool) func(hypothesis []int) float64 {
	return func(hypothesis []int) float64 {
		tokenized := make([]int, len(premise)+len(hypothesis))
		copy(tokenized[0:len(premise)], premise)
		copy(tokenized[len(premise):], hypothesis)

		logits := m.Model.Forward(tokenized)
		if !multiClass {
			return logits.Value().(mat.Matrix).ScalarAt(m.entailmentID).F64()
		}

		// softmax over the entailment vs. contradiction for each label independently
		return mat.NewDense[float64](mat.WithBacking(sliceFromIndices(logits.Value().(mat.Matrix), m.entailmentID, m.contradictionID))).
			Softmax().
			ScalarAt(0).
			F64()
	}
}

// sliceFromIndices returns a new slice containing the elements of the vector at the given indices
func sliceFromIndices(v mat.Matrix, indices ...int) []float64 {
	result := make([]float64, len(indices))
	for i, idx := range indices {
		result[i] = v.ScalarAt(idx).F64()
	}
	return result
}
