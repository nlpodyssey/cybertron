// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Additional copyright notes in the package README.

package generationutils

import (
	"container/heap"
	"sort"

	"github.com/nlpodyssey/cybertron/pkg/utils/sliceutils"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
)

// ScoreProcessor is a function that takes a matrix of scores and returns an altered matrix of scores.
type ScoreProcessor func(scores mat.Matrix) mat.Matrix

// ProcessScores applies a list of score processors to a matrix of scores.
func ProcessScores(processors ...ScoreProcessor) ScoreProcessor {
	return func(scores mat.Matrix) mat.Matrix {
		for _, p := range processors {
			scores = p(scores)
		}
		return scores
	}
}

// TemperatureProcessor applies a temperature to a matrix of scores.
func TemperatureProcessor(temperature float64) ScoreProcessor {
	if temperature == 1 {
		return func(scores mat.Matrix) mat.Matrix {
			return scores
		}
	}
	invTemperature := 1 / temperature
	return func(scores mat.Matrix) mat.Matrix {
		return scores.ProdScalar(invTemperature)
	}
}

// TopKProcessor applies a top-k filter to a matrix of scores.
func TopKProcessor(topK int, filterValue float64) ScoreProcessor {
	return func(scores mat.Matrix) mat.Matrix {
		topK := topK
		if size := scores.Size(); size <= topK {
			topK = size
		}

		inScores := scores.Data().F64()

		rawTopScores := make(sliceutils.OrderedHeap[float64], len(inScores))
		copy(rawTopScores, inScores)

		topScores := sliceutils.ReverseHeap(&rawTopScores)
		heap.Init(topScores)
		for i := 1; i < topK; i++ {
			heap.Pop(topScores)
		}
		minScore := heap.Pop(topScores).(float64)

		return scores.Apply(func(_, _ int, v float64) float64 {
			if v < minScore {
				return filterValue
			}
			return v
		})
	}
}

// TopPProcessor applies a top-p filter to a matrix of scores.
// Note that when using beam decoding (with beam > 1) then minSize must be at least 2.
func TopPProcessor[T float.DType](topP, filterValue T, minSize int) ScoreProcessor {
	return func(scores mat.Matrix) mat.Matrix {
		dataCopy := make([]T, scores.Size())
		copy(dataCopy, mat.Data[T](scores))
		sortedData := sliceutils.NewIndexedSlice[T](dataCopy)
		sort.Stable(sort.Reverse(sortedData))

		cumulativeProbs := mat.NewDense[T](mat.WithBacking(sortedData.Slice)).Softmax().CumSum()
		cumProbData := mat.Data[T](cumulativeProbs)

		indicesToRemove := make([]bool, len(cumProbData))
		for i, cp := range cumProbData {
			indicesToRemove[i] = cp > topP
		}

		if minSize > 1 {
			// Keep at least minSize (minSize-1 because we add the first one below)
			for i := minSize - 1; i >= 0; i-- {
				indicesToRemove[i] = false
			}
		}

		// Shift the indices to the right to keep also the first token above the threshold
		copy(indicesToRemove[1:], indicesToRemove[:len(indicesToRemove)-1])
		indicesToRemove[0] = false

		// Scatter sorted tensors to original indexing

		outData := make([]T, scores.Size())
		copy(outData, mat.Data[T](scores))
		for maskIndex, toRemove := range indicesToRemove {
			if !toRemove {
				continue
			}
			index := sortedData.Indices[maskIndex]
			outData[index] = filterValue
		}

		return mat.NewDense[T](mat.WithBacking(outData))
	}
}
