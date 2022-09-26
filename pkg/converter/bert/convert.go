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
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
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

	// Enable training mode, so that we have writing permissions
	// (for example, for embeddings storage files).
	config.Cybertron.Training = true

	config.Cybertron.TokensStoreName = "tokens"
	config.Cybertron.PositionsStoreName = "positions"
	config.Cybertron.TokenTypesStoreName = "token_types"

	if config.ModelType == "bert" || config.EmbeddingsSize == 0 {
		config.EmbeddingsSize = config.HiddenSize
	}

	pyParams := pytorch.NewParamsProvider[T]().
		WithNameMapping(fixParamsName).
		WithPreProcessing(fixAttentionLayers[T](config))

	if err = pyParams.Load(pyModelFilename); err != nil {
		return err
	}

	repo, err := diskstore.NewRepository(filepath.Join(modelDir, "repo"), diskstore.ReadWriteMode)
	if err != nil {
		panic(err)
	}
	defer func() {
		err = repo.Close()
		if err != nil {
			panic(err)
		}
	}()
	if err := repo.DropAll(); err != nil {
		panic(err)
	}

	baseModel := bert.New[T](config, repo)
	{
		source := pyParams.Pop("bert.embeddings.word_embeddings.weight")
		size := baseModel.Embeddings.Tokens.Config.Size
		for i := 0; i < config.VocabSize; i++ {
			key, _ := vocab.Term(i)
			if len(key) == 0 {
				continue // skip empty key
			}
			item, _ := baseModel.Embeddings.Tokens.Embedding(key)
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

	params := make(paramsMap)
	mapPooler(baseModel.Pooler, params)
	mapEmbeddingsLayerNorm(baseModel.Embeddings.Norm, params)
	mapEncoderParams(baseModel.Encoder, params)

	var finalModel any
	switch config.Architectures[0] {
	case "BertBase":
		finalModel = baseModel
	case "BertForQuestionAnswering":
		qaModel := bert.NewModelForQuestionAnswering[T](baseModel)
		mapQAClassifier(qaModel.Classifier, params)
		finalModel = qaModel

	case "BertForSequenceClassification":
		scModel := bert.NewModelForSequenceClassification[T](baseModel)
		mapSeqClassifier(scModel.Classifier, params)
		finalModel = scModel

	case "BertForTokenClassification":
		tcModel := bert.NewModelForTokenClassification[T](baseModel)
		mapTokenClassifier(tcModel.Classifier, params)
		finalModel = tcModel

	case "HF_ColBERT":
		colbertModel := bert.NewColbertModel[T](baseModel)
		mapLinear(colbertModel.Linear, params)
		finalModel = colbertModel
	default:
		panic(fmt.Errorf("bert: unsupported architecture %s", config.Architectures[0]))
	}

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
	if config.Architectures == nil {
		config.Architectures = append(config.Architectures, "BertBase")
	}

	fmt.Println("Done.")

	return nil
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
