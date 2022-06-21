// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"encoding/gob"
	"sync"

	"github.com/nlpodyssey/cybertron/pkg/generationutils"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/linear"
)

var _ nn.Model = &ModelForConditionalGeneration{}

// ModelForConditionalGeneration is a model for conditional generation tasks
// which embeds a Bart fine-tuned model.
type ModelForConditionalGeneration struct {
	nn.Module
	// Bart is the fine-tuned BART model.
	Bart *Model
	// Projection is the projection layer from the decoder output to the vocabulary.
	Projection *linear.Model
	// PadMask is the mask for the pad token.
	PadMask *nn.Buffer
	// EosMask is the mask for the EOS token.
	EosMask *nn.Buffer
}

func init() {
	gob.Register(&ModelForConditionalGeneration{})
}

// NewModelForConditionalGeneration returns a new model for conditional generation.
func NewModelForConditionalGeneration[T float.DType](bart *Model) *ModelForConditionalGeneration {
	c := bart.Config
	return &ModelForConditionalGeneration{
		Bart:       bart,
		Projection: linear.New[T](c.DModel, c.VocabSize),
		PadMask:    makePadMask[T](c.PadTokenID, c.VocabSize),
		EosMask:    makeEosMask[T](c.EosTokenID, c.VocabSize),
	}
}

// makePadMask returns a mask for padding.
func makePadMask[T float.DType](padTokenID int, vocabSize int) *nn.Buffer {
	mask := mat.NewInitVecDense[T](vocabSize, 0)
	mask.SetVecScalar(padTokenID, float.Interface(mat.Inf[T](-1)))
	return nn.Buf(mask)
}

// makeEosMask returns a mask for EOS.
func makeEosMask[T float.DType](eosTokenID int, vocabSize int) *nn.Buffer {
	mask := mat.NewInitVecDense[T](vocabSize, mat.Inf[T](-1))
	mask.SetVecScalar(eosTokenID, float.Interface(T(0)))
	return nn.Buf(mask)
}

// DecodingInput is the input for the decoding function of the model for conditional generation.
type DecodingInput struct {
	// InputIDs are the input IDs for the decoder.
	InputIDs []int
	// CurLen is the current length of the generating sequence.
	CurLen int
	// Cache is the cache for the decoder.
	Cache Cache
}

// DecodingOutput is the output of the decoding function of the model for conditional generation.
type DecodingOutput struct {
	// LogProbRaw is the raw (not processed) log probability of the generated token.
	LogProbRaw ag.Node
	// LogProbValue is the post-processed log probability of the generated token.
	LogProbValue mat.Matrix
	// NextCache is the next cache.
	NextCache Cache
}

// DecodingFunc returns a decoding function that works using the encoder states derived from the input.
// During inference, it adjusts the logits to avoid impossible tokens.
func (m *ModelForConditionalGeneration) DecodingFunc(encoderInputIDs []int, scoreProc generationutils.ScoreProcessor, inference bool) func(batch []*DecodingInput) []*DecodingOutput {
	encoderStates := m.Bart.Encoder.Encode(encoderInputIDs)

	return func(batch []*DecodingInput) []*DecodingOutput {
		result := make([]*DecodingOutput, len(batch))

		var wg sync.WaitGroup
		wg.Add(len(batch))

		for i, item := range batch {
			i, item := i, item
			go func() {
				defer wg.Done()
				result[i] = m.next(decodingState{
					encoderStates: encoderStates,
					decodingInput: item,
					scoreProc:     scoreProc,
					inference:     inference,
				})
			}()
		}
		wg.Wait()
		return result
	}
}

// decodingState is a state for the decoding function.
type decodingState struct {
	encoderStates []ag.Node
	decodingInput *DecodingInput
	scoreProc     generationutils.ScoreProcessor
	inference     bool
}

// next returns the post-processed log probability for the generated tokens.
func (m *ModelForConditionalGeneration) next(state decodingState) *DecodingOutput {
	decoded, nextCache := m.Bart.Decoder.Decode(
		state.encoderStates,
		state.decodingInput.InputIDs,
		state.decodingInput.Cache,
		state.decodingInput.CurLen,
	)

	logits := m.Projection.Forward(decoded...)[0]
	if state.inference {
		logits = m.adjustLogits(logits, state.decodingInput.CurLen)
	}

	logProb := ag.LogSoftmax(logits)

	return &DecodingOutput{
		LogProbRaw:   logProb,
		LogProbValue: state.scoreProc(logProb.Value()),
		NextCache:    nextCache,
	}
}

// adjustLogits applies the mask to the logits to avoid impossible token from being generated during inference.
func (m *ModelForConditionalGeneration) adjustLogits(xs ag.Node, curLen int) ag.Node {
	ys := ag.Add(xs, m.PadMask) // Don't generate pad token
	if curLen == m.Bart.Config.MaxLength-1 && m.Bart.Config.EosTokenID >= 0 {
		ys = ag.Add(ys, m.EosMask) // Force EOS to be generated
	}
	return ys
}
