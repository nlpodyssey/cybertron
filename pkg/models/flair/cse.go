// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"encoding/gob"
	"strings"
	"sync"

	"github.com/nlpodyssey/cybertron/pkg/models/flair/charlm"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/nn"
)

var _ nn.Model = &ContextualStringEmbeddings{}

// MergeType is the enumeration-like type used for the set of merging methods used by the ContextualStringEmbeddings.
type MergeType int

const (
	// Concat merging method: the outputs are concatenated together (the default)
	Concat MergeType = iota
	// Sum merging method: the outputs are added together
	Sum
	// Prod merging method: the outputs multiplied element-wise together
	Prod
	// Avg merging method: the average of the outputs is taken
	Avg
)

type ContextualStringEmbeddings struct {
	nn.Module
	LeftToRight *charlm.Model
	RightToLeft *charlm.Model
	MergeMode   MergeType
	StartMarker rune
	EndMarker   rune
}

func init() {
	gob.Register(&ContextualStringEmbeddings{})
}

func NewContextualStringEmbeddings(leftToRight, rightToLeft *charlm.Model, merge MergeType, startMarker, endMarker rune) *ContextualStringEmbeddings {
	return &ContextualStringEmbeddings{
		LeftToRight: leftToRight,
		RightToLeft: rightToLeft,
		MergeMode:   merge,
		StartMarker: startMarker,
		EndMarker:   endMarker,
	}
}

type text struct {
	string
	tokens []string
}

// EncodeTokens performs the forward step for each input and returns the result.
func (m *ContextualStringEmbeddings) EncodeTokens(tokens []string) []mat.Tensor {
	t := text{
		string: strings.Join(tokens, " "),
		tokens: tokens,
	}

	h, rh := m.computeHiddenStates(chars(t.string))

	result := make([]mat.Tensor, len(t.tokens))
	for i, boundary := range t.boundaries() {
		result[i] = m.merge(rh[boundary[1]], h[boundary[0]])
	}
	return result
}

func (m *ContextualStringEmbeddings) computeHiddenStates(sequence []string) (hiddenStates, rHiddenStates []mat.Tensor) {
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		hiddenStates = m.LeftToRight.EncodeTokens(pad(sequence, m.StartMarker, m.EndMarker))
		wg.Done()
	}()

	go func() {
		rHiddenStates = m.RightToLeft.EncodeTokens(pad(reversed(sequence), m.StartMarker, m.EndMarker))
		wg.Done()
	}()
	wg.Wait()
	return
}

func (m *ContextualStringEmbeddings) merge(a, b mat.Tensor) mat.Tensor {
	switch m.MergeMode {
	case Concat:
		return ag.Concat(a, b)
	case Sum:
		return ag.Add(a, b)
	case Prod:
		return ag.Prod(a, b)
	case Avg:
		return ag.ProdScalar(ag.Add(a, b), mat.Scalar(0.5))
	default:
		panic("flair: invalid merge mode for the ContextualStringEmbeddings")
	}
}

func (t *text) boundaries() [][2]int {
	textLength := len([]rune(t.string))
	boundaries := make([][2]int, len(t.tokens))
	start := 0
	for i, word := range t.tokens {
		wordLength := len([]rune(word))
		boundaries[i] = [2]int{
			start + wordLength + 1, // end index
			textLength - start + 1, // reverse end index
		}
		start += wordLength + 1
	}
	return boundaries
}

func reversed(words []string) []string {
	r := make([]string, len(words))
	copy(r, words)
	for i := 0; i < len(r)/2; i++ {
		j := len(r) - i - 1
		r[i], r[j] = r[j], r[i]
	}
	return r
}

func pad(sequence []string, startMarker, endMarker rune) []string {
	length := len(sequence) + 2
	padded := make([]string, length)
	padded[0] = string(startMarker)
	padded[length-1] = string(endMarker)
	copy(padded[1:length-1], sequence)
	return padded
}

func chars(text string) []string {
	result := make([]string, 0)
	for _, item := range text {
		result = append(result, string(item))
	}
	return result
}
