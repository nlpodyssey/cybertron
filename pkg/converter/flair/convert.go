// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
	convflair "github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/flair"
	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/torch"
	"github.com/nlpodyssey/cybertron/pkg/models/flair"
	"github.com/nlpodyssey/cybertron/pkg/models/flair/charlm"
	"github.com/nlpodyssey/cybertron/pkg/vocabulary"
	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/birnn"
	"github.com/nlpodyssey/spago/nn/crf"
	"github.com/nlpodyssey/spago/nn/embedding"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/nlpodyssey/spago/nn/recurrent/lstm"
	"github.com/rs/zerolog/log"
)

const (
	// defaultPyModelFilename is the default Flair PyTorch model filename.
	defaultPyModelFilename = "pytorch_model.bin"
	// defaultGoModelFilename is the default Flair spaGO model filename.
	defaultGoModelFilename = "spago_model.bin"
	// defaultConfigFilename is the default Flair JSON configuration filename.
	defaultConfigFilename = "config.json"
)

// Convert converts a Flair PyTorch model to a Spago/Cybertron model.
func Convert[T float.DType](modelDir string, overwriteIfExist bool) (err error) {
	pyModelFilename := filepath.Join(modelDir, defaultPyModelFilename)
	goModelFilename := filepath.Join(modelDir, defaultGoModelFilename)
	configFilename := filepath.Join(modelDir, defaultConfigFilename)

	if info, err := os.Stat(goModelFilename); !overwriteIfExist && err == nil && !info.IsDir() {
		log.Info().Str("model", goModelFilename).Msg("model file already exists, skipping conversion")
		return nil
	}

	st, err := convflair.LoadSequenceTagger(pyModelFilename)
	if err != nil {
		return fmt.Errorf("failed to load sequence tagger: %w", err)
	}

	conv := &converter[T]{
		st: st,
	}

	err = createConfigFile(conv.config(), configFilename)
	if err != nil {
		return fmt.Errorf("failed to create flair config file: %w", err)
	}

	m, err := conv.flairModel()
	if err != nil {
		return fmt.Errorf("failed to convert flair Model: %w", err)
	}

	err = nn.DumpToFile(m, goModelFilename)
	if err != nil {
		return fmt.Errorf("failed to dump converted model to file: %w", err)
	}

	return nil
}

func createConfigFile(conf flair.Config, filename string) (err error) {
	f, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("failed to create config file %q: %w", filename, err)
	}
	defer func() {
		if e := f.Close(); e != nil && err == nil {
			err = fmt.Errorf("failed to close config file %q: %w", filename, e)
		}
	}()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	enc.SetEscapeHTML(false)

	err = enc.Encode(conf)
	if err != nil {
		return fmt.Errorf("failed to encode JSON config: %w", err)
	}

	return nil
}

type converter[T float.DType] struct {
	st *convflair.SequenceTagger
}

func (conv *converter[T]) flairModel() (m *flair.Model, err error) {
	m = new(flair.Model)
	m.Encoder, err = conv.encoder()
	if err != nil {
		return nil, fmt.Errorf("failed to convert Encoder: %w", err)
	}
	m.Decoder, err = conv.decoder()
	if err != nil {
		return nil, fmt.Errorf("failed to convert Decoder: %w", err)
	}
	return m, nil
}

func (conv *converter[T]) encoder() (e *flair.Encoder, err error) {
	e = new(flair.Encoder)
	e.Embeddings, err = conv.encoderEmbeddings()
	if err != nil {
		return nil, fmt.Errorf("failed to convert Embeddings: %w", err)
	}
	e.BiRNN, err = conv.encoderBiRNN()
	if err != nil {
		return nil, fmt.Errorf("failed to convert BiRNN: %w", err)
	}
	return e, nil
}

func (conv *converter[T]) encoderEmbeddings() (e *flair.Embeddings, err error) {
	e = new(flair.Embeddings)
	e.TokensEncoder, err = conv.encoderEmbeddingsTokensEncoder()
	if err != nil {
		return nil, fmt.Errorf("failed to convert TokensEncoder: %w", err)
	}
	e.Projection, err = conv.convertLinear(conv.st.Embedding2NN)
	if err != nil {
		return nil, fmt.Errorf("failed to convert Projection: %w", err)
	}
	return e, nil
}

func (conv *converter[T]) encoderEmbeddingsTokensEncoder() ([]flair.TokensEncoder, error) {
	se, err := conversion.AssertType[*convflair.StackedEmbeddings](conv.st.Embeddings)
	if err != nil {
		return nil, fmt.Errorf("unexpected SequenceTagger.Embeddings: %w", err)
	}

	var flairWe []*convflair.WordEmbeddings
	var flairEmbForward *convflair.FlairEmbeddings
	var flairEmbBackward *convflair.FlairEmbeddings

	for _, item := range se.Embeddings {
		switch tt := item.(type) {
		case *convflair.WordEmbeddings:
			flairWe = append(flairWe, tt)
		case *convflair.FlairEmbeddings:
			if tt.IsForwardLm {
				if flairEmbForward != nil {
					return nil, fmt.Errorf("more than one FlairEmbeddings-forward found in StackedEmbeddings")
				}
				flairEmbForward = tt
			} else {
				if flairEmbBackward != nil {
					return nil, fmt.Errorf("more than one FlairEmbeddings-backward found in StackedEmbeddings")
				}
				flairEmbBackward = tt
			}
		default:
			return nil, fmt.Errorf("unexpected TokenEmbeddings type: %T", item)
		}
	}

	switch {
	case flairEmbForward == nil:
		return nil, fmt.Errorf("FlairEmbeddings-forward not found in StackedEmbeddings")
	case flairEmbBackward == nil:
		return nil, fmt.Errorf("FlairEmbeddings-backward not found in StackedEmbeddings")
	}

	tEnc := make([]flair.TokensEncoder, 0)

	{ // --- Word embeddings ---
		for _, item := range flairWe {
			we, err := conv.encoderEmbeddingsTokensEncoderWordEmbeddings(item)
			if err != nil {
				return nil, fmt.Errorf("failed to convert WordEmbeddings: %w", err)
			}
			_ = we
			//tEnc = append(tEnc, we) // TODO: create a wrapper of embedding.Model adding vocabulary and implement TokensEncoder
		}
	}

	{ // --- CharLM embeddings ---
		voc, err := conv.encoderEmbeddingsTokensEncoderCharLMVocabulary(flairEmbForward, flairEmbBackward)
		if err != nil {
			return nil, fmt.Errorf("failed to convert FlairEmbeddings CharLM vocabulary: %w", err)
		}

		l2r, err := conv.encoderEmbeddingsTokensEncoderCharLM(voc, flairEmbForward)
		if err != nil {
			return nil, fmt.Errorf("failed to convert FlairEmbeddings-forward: %w", err)
		}
		r2l, err := conv.encoderEmbeddingsTokensEncoderCharLM(voc, flairEmbBackward)
		if err != nil {
			return nil, fmt.Errorf("failed to convert FlairEmbeddings-backward: %w", err)
		}
		tEnc = append(tEnc, &flair.ContextualStringEmbeddings{
			LeftToRight: l2r,
			RightToLeft: r2l,
			MergeMode:   flair.Concat,
			StartMarker: '\n',
			EndMarker:   ' ',
		})
	}

	return tEnc, nil
}

func (conv *converter[T]) encoderEmbeddingsTokensEncoderWordEmbeddings(we *convflair.WordEmbeddings) (*flair.WordEmbeddings, error) {

	emb := embedding.New[T](len(we.Vocab), we.Embedding.EmbeddingDim)

	weights := we.Embedding.Weight
	for key, index := range we.Vocab {
		_ = key // TODO: use key
		emb.Weights[index].ReplaceValue(weights[index])
	}

	return &flair.WordEmbeddings{Model: emb}, nil
}

func (conv *converter[T]) encoderEmbeddingsTokensEncoderCharLMVocabulary(forward, backward *convflair.FlairEmbeddings) (*vocabulary.Vocabulary, error) {
	d1 := forward.LM.Dictionary
	d2 := backward.LM.Dictionary
	if !reflect.DeepEqual(d1.Idx2Item, d2.Idx2Item) {
		return nil, fmt.Errorf("FlairEmbeddings LM forward/backward dictionaries differ")
	}
	return vocabulary.New(d1.Idx2Item), nil
}

func (conv *converter[T]) encoderEmbeddingsTokensEncoderCharLM(voc *vocabulary.Vocabulary, fe *convflair.FlairEmbeddings) (*charlm.Model, error) {
	decoder, err := conv.convertLinear(fe.LM.Decoder)
	if err != nil {
		return nil, fmt.Errorf("failed to convert CharLM decoder: %w", err)
	}

	rnn, err := conv.convertLSTM(fe.LM.RNN, false)
	if err != nil {
		return nil, fmt.Errorf("failed to convert CharLM RNN: %w", err)
	}

	emb, err := conv.encoderEmbeddingsTokensEncoderCharLMEmbeddings(voc, fe)
	if err != nil {
		return nil, fmt.Errorf("failed to convert CharLM embeddings: %w", err)
	}

	var proj *linear.Model
	if fe.LM.Proj != nil {
		proj, err = conv.convertLinear(fe.LM.Proj)
		if err != nil {
			return nil, fmt.Errorf("failed to convert CharLM proj: %w", err)
		}
	}

	return &charlm.Model{
		Config: charlm.Config{
			Name:              fe.Name,
			VocabularySize:    voc.Size(),
			EmbeddingSize:     fe.LM.EmbeddingSize,
			HiddenSize:        fe.LM.HiddenSize,
			OutputSize:        fe.LM.NOut,
			SequenceSeparator: "\n",    // TODO: can obtain from fe? Const?
			UnknownToken:      "<unk>", // TODO: can obtain from fe? First from voc? Const?
			Trainable:         false,
		},
		Decoder:    decoder,
		Projection: proj,
		RNN:        rnn,
		Embeddings: emb,
		Vocabulary: voc,
	}, nil
}

func (conv *converter[T]) encoderEmbeddingsTokensEncoderCharLMEmbeddings(voc *vocabulary.Vocabulary, fe *convflair.FlairEmbeddings) (*embedding.Model, error) {
	enc := fe.LM.Encoder

	emb := embedding.New[T](voc.Size(), enc.EmbeddingDim)

	weights := enc.Weight
	for key, index := range voc.Map() {
		_ = key
		// TODO: key
		emb.Weights[index].ReplaceValue(weights[index])
	}

	return emb, nil
}

func (conv *converter[T]) convertLinear(tl *torch.Linear) (*linear.Model, error) {
	w, err := conv.TensorToMatrix(tl.Weight)
	if err != nil {
		return nil, fmt.Errorf("failed to convert linear weights: %w", err)
	}
	b, err := conv.TensorToVector(tl.Bias)
	if err != nil {
		return nil, fmt.Errorf("failed to convert linear bias: %w", err)
	}
	return &linear.Model{
		W: nn.NewParam(w),
		B: nn.NewParam(b),
	}, nil
}

func (conv *converter[T]) convertLSTM(tl *torch.LSTM, reverse bool) (*lstm.Model, error) {
	params, err := conv.extractLSTMParams(tl, reverse)
	if err != nil {
		return nil, fmt.Errorf("failed to extract LSTM params: %w", err)
	}

	wih, err := splitMatrixInto4(params.wih)
	if err != nil {
		return nil, fmt.Errorf("failed to split weight-ih matrix into 4 parts: %w", err)
	}
	whh, err := splitMatrixInto4(params.whh)
	if err != nil {
		return nil, fmt.Errorf("failed to split weight-hh matrix into 4 parts: %w", err)
	}
	bih, err := splitMatrixInto4(params.bih)
	if err != nil {
		return nil, fmt.Errorf("failed to split bias-ih matrix into 4 parts: %w", err)
	}
	bhh, err := splitMatrixInto4(params.bhh)
	if err != nil {
		return nil, fmt.Errorf("failed to split bias-hh matrix into 4 parts: %w", err)
	}

	var b [4]mat.Matrix
	for i := range b {
		b[i] = bih[i].Add(bhh[i])
	}

	return &lstm.Model{
		UseRefinedGates: false,

		WIn:    nn.NewParam(wih[0]),
		WInRec: nn.NewParam(whh[0]),
		BIn:    nn.NewParam(b[0]),

		WFor:    nn.NewParam(wih[1]),
		WForRec: nn.NewParam(whh[1]),
		BFor:    nn.NewParam(b[1]),

		WCand:    nn.NewParam(wih[2]),
		WCandRec: nn.NewParam(whh[2]),
		BCand:    nn.NewParam(b[2]),

		WOut:    nn.NewParam(wih[3]),
		WOutRec: nn.NewParam(whh[3]),
		BOut:    nn.NewParam(b[3]),
	}, nil
}

type lstmParams struct {
	wih mat.Matrix
	whh mat.Matrix
	bih mat.Matrix
	bhh mat.Matrix
}

func (conv *converter[T]) extractLSTMParams(tl *torch.LSTM, reverse bool) (lps lstmParams, err error) {
	lps.wih, err = conv.extractLSTMParam(tl, "weight_ih_l0", reverse)
	if err != nil {
		return
	}
	lps.whh, err = conv.extractLSTMParam(tl, "weight_hh_l0", reverse)
	if err != nil {
		return
	}
	lps.bih, err = conv.extractLSTMParam(tl, "bias_ih_l0", reverse)
	if err != nil {
		return
	}
	lps.bhh, err = conv.extractLSTMParam(tl, "bias_hh_l0", reverse)
	if err != nil {
		return
	}
	return
}

func (conv *converter[T]) extractLSTMParam(tl *torch.LSTM, name string, reverse bool) (lp mat.Matrix, err error) {
	if reverse {
		name += "_reverse"
	}

	tp, ok := tl.Parameters[name]
	if !ok {
		return nil, fmt.Errorf("LSTM parameter %q not found", name)
	}

	switch {
	case strings.HasPrefix(name, "weight"):
		lp, err = conv.TensorToMatrix(tp.Data)
	case strings.HasPrefix(name, "bias"):
		lp, err = conv.TensorToVector(tp.Data)
	default:
		err = fmt.Errorf("malformed or unimplemented parameter name")
	}
	if err != nil {
		return nil, fmt.Errorf("failed to convert LSTM parameter %q: %w", name, err)
	}
	return lp, err
}

func splitMatrixInto4(m mat.Matrix) (parts [4]mat.Matrix, _ error) {
	rows, cols := m.Dims()

	if rows == 0 || rows%4 != 0 {
		return parts, fmt.Errorf("cannot split matrix with %d rows into 4 parts", rows)
	}

	partRows := rows / 4
	for i := range parts {
		parts[i] = m.Slice(partRows*i, 0, partRows*(i+1), cols)
	}
	return parts, nil
}

func (conv *converter[T]) encoderBiRNN() (*birnn.Model, error) {
	tl, ok := conv.st.RNN.(*torch.LSTM)
	if !ok {
		return nil, fmt.Errorf("want LSTM, got %T", conv.st.RNN)
	}

	positive, err := conv.convertLSTM(tl, false)
	if err != nil {
		return nil, fmt.Errorf("failed to convert BiRNN positive LSTM: %w", err)
	}
	negative, err := conv.convertLSTM(tl, true)
	if err != nil {
		return nil, fmt.Errorf("failed to convert BiRNN negative LSTM: %w", err)
	}

	return &birnn.Model{
		Positive:  positive,
		Negative:  negative,
		MergeMode: birnn.Concat, // TODO: verify
	}, nil
}

func (conv *converter[T]) decoder() (_ *flair.Decoder, err error) {
	d := new(flair.Decoder)
	d.Scorer, err = conv.convertLinear(conv.st.Linear)
	if err != nil {
		return nil, fmt.Errorf("failed to convert Scorer: %w", err)
	}
	d.CRF, err = conv.decoderCRF()
	if err != nil {
		return nil, fmt.Errorf("failed to convert CRF: %w", err)
	}
	return d, nil
}

func (conv *converter[T]) decoderCRF() (*crf.Model, error) {
	scores, err := conv.convTransitions()
	if err != nil {
		return nil, err
	}
	return &crf.Model{
		Size:             conv.st.CRF.TagsetSize, // TODO: remove "start" and "end" (?)
		TransitionScores: nn.NewParam(scores),
	}, nil
}

func (conv *converter[T]) convTransitions() (mat.Matrix, error) {
	labelsSize := conv.st.CRF.TagsetSize
	weights, err := conversion.GetTensorData(conv.st.CRF.Transitions)
	if err != nil {
		return nil, err
	}
	// TODO: is it guaranteed that "<START>" and "<STOP>" are always the last two items?
	startIndex := labelsSize - 2
	stopIndex := labelsSize - 1

	length := labelsSize - 1
	out := make([]T, length*length)
	for i := range out {
		out[i] = -10000
	}

	i := 0

	for rowIndex := 0; rowIndex < labelsSize; rowIndex++ {
		row := weights[rowIndex*labelsSize : rowIndex*labelsSize+labelsSize]
		j := 0
		if i != startIndex { // skip transition ending in start
			if i < startIndex {
				for _, col := range row {
					if j != stopIndex { // skip transition starting in end
						if j == startIndex { // transition starting at start
							out[i+1] = T(col)
						} else {
							out[(j+1)*length+(i+1)] = T(col)
						}
					}
					j++
				}
			} else {
				for _, col := range row {
					if j != stopIndex { // skip transition starting in end
						if j == startIndex { // transition starting at start
							out[0] = T(col)
						} else {
							out[(j+1)*length] = T(col)
						}
					}
					j++
				}
			}
		}
		i++
	}

	return mat.NewDense(length, length, out), nil
}

func (conv *converter[T]) config() flair.Config {
	id2labels := make(map[string]string)
	for id, label := range conv.st.LabelDictionary.Idx2Item {
		id2labels[fmt.Sprintf("%d", id)] = label
	}
	return flair.Config{
		ModelType: "flair",
		ID2Label:  id2labels,
	}
}

func (conv *converter[T]) TensorToMatrix(t *pytorch.Tensor) (*mat.Dense[T], error) {
	if len(t.Size) != 2 {
		return nil, fmt.Errorf("failed to convert tensor to matrix: want 2 dimensions, got %d", len(t.Size))
	}
	data, err := conversion.GetTensorData(t)
	if err != nil {
		return nil, err
	}
	return mat.NewDense[T](t.Size[0], t.Size[1], float.SliceValueOf[T](float.SliceInterface(data))), nil
}

func (conv *converter[T]) TensorToVector(t *pytorch.Tensor) (*mat.Dense[T], error) {
	if len(t.Size) != 1 {
		return nil, fmt.Errorf("failed to convert tensor to vector: want 1 dimension, got %d", len(t.Size))
	}
	data, err := conversion.GetTensorData(t)
	if err != nil {
		return nil, err
	}
	return mat.NewVecDense[T](float.SliceValueOf[T](float.SliceInterface(data))), nil
}
