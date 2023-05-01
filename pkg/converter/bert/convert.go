// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bert

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/converter/pytorch"
	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/vocabulary"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

const (
	// defaultConfigFilename is the default Bart JSON configuration filename.
	defaultConfigFilename = "config.json"
	// defaultVocabularyFile is the default BERT model's vocabulary filename.
	defaultVocabularyFile = "vocab.txt"
	// defaultPyModelFilename is the default Bart PyTorch model filename.
	defaultPyModelFilename = "pytorch_model.bin"
	// defaultGoModelFilename is the default Bart spaGO model filename.
	defaultGoModelFilename = "spago_model.bin"
)

// mappingParam is a mapping between a Hugging Face Transformers parameters and Cybertron parameters.
type mappingParam struct {
	value   mat.Matrix
	matched bool
}

// Convert converts a Bert PyTorch model to a Spago (Cybertron) model.
func Convert[T float.DType](modelDir string, overwriteIfExist bool) error {
	var (
		configFilename  = filepath.Join(modelDir, defaultConfigFilename)
		pyModelFilename = filepath.Join(modelDir, defaultPyModelFilename)
		goModelFilename = filepath.Join(modelDir, defaultGoModelFilename)
		vocabFilename   = filepath.Join(modelDir, defaultVocabularyFile)
	)

	if info, err := os.Stat(goModelFilename); !overwriteIfExist && err == nil && !info.IsDir() {
		log.Info().Str("model", goModelFilename).Msg("model file already exists, skipping conversion")
		return nil
	}

	config, err := bert.ConfigFromFile[bert.Config](configFilename)
	if err != nil {
		return err
	}

	vocab, err := vocabulary.NewFromFile(vocabFilename)
	if err != nil {
		return err
	}

	{
		// Enable training mode, so that we have writing permissions
		// (for example, for embeddings storage files).
		config.Cybertron.Training = true

		if config.ModelType == "bert" || config.EmbeddingsSize == 0 {
			config.EmbeddingsSize = config.HiddenSize
		}
	}

	pyParams := pytorch.NewParamsProvider[T]().
		WithNameMapping(fixParamsName).
		WithPreProcessing(fixAttentionLayers[T](config))

	if err = pyParams.Load(pyModelFilename); err != nil {
		return err
	}

	params := make(paramsMap)
	baseModel := mapBaseModel[T](config, pyParams, params, vocab)
	finalModel := mapSpecificArchitecture[T](baseModel, config.Architectures, params)

	mapping := make(map[string]*mappingParam)
	for k, v := range params {
		mapping[k] = &mappingParam{value: v, matched: false}
	}

	err = pyParams.Iterate(func(name string, value []T) error {
		param, ok := mapping[name]
		if !ok {
			return nil
		}
		if param.value.Size() != len(value) {
			return fmt.Errorf("error setting %s: dim mismatch", name)
		}
		mat.SetData[T](param.value, value)
		param.matched = true
		return nil
	})
	if err != nil {
		return err
	}

	if zerolog.GlobalLevel() <= zerolog.DebugLevel {
		log.Debug().Msg("Reporting possible conversion mapping anomalies")
		for key, value := range mapping {
			if !value.matched {
				log.Debug().Str("parameter", key).Msg("parameter not initialized")
			}
		}
		err = pyParams.Iterate(func(name string, _ []T) error {
			if _, ok := mapping[name]; !ok {
				log.Debug().Str("parameter", name).Msg("parameter not mapped")
			}
			return nil
		})
		if err != nil {
			return err
		}
	}

	fmt.Printf("Serializing model to \"%s\"... ", goModelFilename)
	err = nn.DumpToFile(finalModel, goModelFilename)
	if err != nil {
		return err
	}

	fmt.Println("Done.")

	return nil
}

func mapBaseModel[T float.DType](config bert.Config, pyParams *pytorch.ParamsProvider[T], params paramsMap, vocab *vocabulary.Vocabulary) *bert.Model {
	baseModel := bert.New[T](config)

	{
		source := pyParams.Pop("bert.embeddings.word_embeddings.weight")
		size := baseModel.Embeddings.Tokens.Dim
		for i := 0; i < config.VocabSize; i++ {
			key, _ := vocab.Term(i)
			if len(key) == 0 {
				continue // skip empty key
			}
			item, _ := baseModel.Embeddings.Tokens.Embedding(i)
			item.ReplaceValue(mat.NewVecDense[T](source[i*size : (i+1)*size]))
		}
	}

	cols := config.HiddenSize

	{
		source := pyParams.Pop("bert.embeddings.position_embeddings.weight")
		dest := baseModel.Embeddings.Positions
		for i := 0; i < config.MaxPositionEmbeddings; i++ {
			item, _ := dest.Embedding(i)
			item.ReplaceValue(mat.NewVecDense[T](source[i*cols : (i+1)*cols]))
		}
	}

	{
		source := pyParams.Pop("bert.embeddings.token_type_embeddings.weight")
		dest := baseModel.Embeddings.TokenTypes
		for i := 0; i < config.TypeVocabSize; i++ {
			item, _ := dest.Embedding(i)
			item.ReplaceValue(mat.NewVecDense[T](source[i*cols : (i+1)*cols]))
		}
	}

	mapPooler(baseModel.Pooler, params)
	mapEmbeddingsLayerNorm(baseModel.Embeddings.Norm, params)
	mapEncoderParams(baseModel.Encoder, params)

	return baseModel
}

func mapSpecificArchitecture[T float.DType](baseModel *bert.Model, architectures []string, params paramsMap) nn.Model {
	if architectures == nil {
		architectures = append(architectures, "BertBase")
	}

	switch architectures[0] {
	case "BertBase":
		return baseModel
	case "BertModel":
		return bert.NewModelForSequenceEncoding(baseModel)
	case "BertForMaskedLM":
		m := bert.NewModelForMaskedLM[T](baseModel)
		mapMaskedLM(m.Layers, params)
		return m
	case "BertForQuestionAnswering":
		m := bert.NewModelForQuestionAnswering[T](baseModel)
		mapQAClassifier(m.Classifier, params)
		return m
	case "BertForSequenceClassification":
		m := bert.NewModelForSequenceClassification[T](baseModel)
		mapSeqClassifier(m.Classifier, params)
		return m
	case "BertForTokenClassification":
		m := bert.NewModelForTokenClassification[T](baseModel)
		mapTokenClassifier(m.Classifier, params)
		return m
	default:
		panic(fmt.Errorf("bert: unsupported architecture %s", architectures[0]))
	}
}

func fixParamsName(from string) (to string) {
	to = from
	to = strings.Replace(to, "electra.", "bert.", -1)
	to = strings.Replace(to, ".gamma", ".weight", -1)
	to = strings.Replace(to, ".beta", ".bias", -1)
	if strings.HasPrefix(to, "embeddings.") {
		to = fmt.Sprintf("bert.%s", to)
	}
	if strings.HasPrefix(to, "encoder.") {
		to = fmt.Sprintf("bert.%s", to)
	}
	if strings.HasPrefix(to, "pooler.") {
		to = fmt.Sprintf("bert.%s", to)
	}
	return
}
