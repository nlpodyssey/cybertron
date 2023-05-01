// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/converter/pytorch"
	"github.com/nlpodyssey/cybertron/pkg/models/bart"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

const (
	// defaultConfigFilename is the default Bart JSON configuration filename.
	defaultConfigFilename = "config.json"
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

// Convert converts a Bart PyTorch model to a Spago (Cybertron) model.
func Convert[T float.DType](modelDir string, overwriteIfExist bool) error {
	configFilename := filepath.Join(modelDir, defaultConfigFilename)
	pyModelFilename := filepath.Join(modelDir, defaultPyModelFilename)
	goModelFilename := filepath.Join(modelDir, defaultGoModelFilename)

	if info, err := os.Stat(goModelFilename); !overwriteIfExist && err == nil && !info.IsDir() {
		log.Info().Str("model", goModelFilename).Msg("model file already exists, skipping conversion")
		return nil
	}

	config, err := bart.ConfigFromFile(configFilename)
	if err != nil {
		return err
	}

	// Enable training mode, so that we have writing permissions
	// (for example, for embeddings storage files).
	config.Cybertron.Training = true

	// Bart-specific configuration for spaGO
	if config.ModelType == "bart" {
		// This offsets takes into account Bart's padding.
		// Its value is left zero for other model, such as Marian.
		config.Cybertron.PositionalEncoderOffset = 2
	}

	pyParams := pytorch.NewParamsProvider[T]().
		WithNameMapping(fixParamsName).
		WithPreProcessing(fixAttentionLayers[T](config))

	if err = pyParams.Load(pyModelFilename); err != nil {
		return err
	}

	m := bart.New[T](config)
	bartForConditionalGenertion := bart.NewModelForConditionalGeneration[T](m)
	bartForSequenceClassification := bart.NewModelForSequenceClassification[T](m)
	{
		source := pyParams.Get("model.shared.weight")
		size := m.Embeddings.Dim
		for i := 0; i < config.VocabSize; i++ {
			item, _ := m.Embeddings.Embedding(i)
			item.ReplaceValue(mat.NewVecDense[T](source[i*size : (i+1)*size]))
		}
	}

	if !config.StaticPositionEmbeddings {
		rows := config.MaxPositionEmbeddings + config.Cybertron.PositionalEncoderOffset
		cols := config.DModel

		{
			source := pyParams.Get("model.encoder.embed_positions.weight")
			dest := m.Encoder.Embeddings.PositionalEncoder.Embeddings
			for i := 0; i < rows; i++ {
				item, _ := dest.Embedding(i)
				item.ReplaceValue(mat.NewVecDense[T](source[i*cols : (i+1)*cols]))
			}
		}

		{
			source := pyParams.Get("model.decoder.embed_positions.weight")
			dest := m.Decoder.Embeddings.PositionalEncoder.Embeddings
			for i := 0; i < rows; i++ {
				item, _ := dest.Embedding(i)
				item.ReplaceValue(mat.NewVecDense[T](source[i*cols : (i+1)*cols]))
			}
		}
	}

	params := make(paramsMap)
	mapEncoderParams(m.Encoder, params)
	mapDecoderParams(m.Decoder, params)
	mapClassifier(bartForSequenceClassification.Classifier, params)
	mapProjectionLayer(bartForConditionalGenertion.Projection, params)

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

	if zerolog.GlobalLevel() < zerolog.DebugLevel {
		log.Trace().Msg("Reporting possible conversion mapping anomalies")
		for key, value := range mapping {
			if !value.matched {
				log.Trace().Str("parameter", key).Msg("parameter not initialized")
			}
		}
		err = pyParams.Iterate(func(name string, _ []T) error {
			if _, ok := mapping[name]; !ok {
				log.Trace().Str("parameter", name).Msg("parameter not mapped")
			}
			return nil
		})
		if err != nil {
			return err
		}
	}

	fmt.Printf("Serializing model to \"%s\"... ", goModelFilename)
	if config.Architecture == nil {
		config.Architecture = append(config.Architecture, "BartBase")
	}

	{
		switch config.Architecture[0] {
		case "BartBase":
			err := nn.DumpToFile(m, goModelFilename)
			if err != nil {
				return err
			}
		case "BartForSequenceClassification":
			err := nn.DumpToFile(bartForSequenceClassification, goModelFilename)
			if err != nil {
				return err
			}
		case "MarianMTModel", "PegasusForConditionalGeneration", "BartForConditionalGeneration":
			err := nn.DumpToFile(bartForConditionalGenertion, goModelFilename)
			if err != nil {
				return err
			}
		default:
			panic(fmt.Errorf("bart: unsupported architecture %s", config.Architecture[0]))
		}
	}

	fmt.Println("Done.")

	return nil
}

func fixParamsName(from string) (to string) {
	to = from
	to = strings.Replace(to, ".gamma", ".weight", -1)
	to = strings.Replace(to, ".beta", ".bias", -1)
	return
}
