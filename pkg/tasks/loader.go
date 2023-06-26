// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tasks

import (
	"errors"
	"fmt"
	"reflect"

	"github.com/nlpodyssey/cybertron/pkg/converter"
	"github.com/nlpodyssey/cybertron/pkg/downloader"
	"github.com/nlpodyssey/cybertron/pkg/models"
	"github.com/nlpodyssey/cybertron/pkg/tasks/languagemodeling"
	bert_for_language_modeling "github.com/nlpodyssey/cybertron/pkg/tasks/languagemodeling/bert"
	distilbert_for_language_modeling "github.com/nlpodyssey/cybertron/pkg/tasks/languagemodeling/distilbert"
	"github.com/nlpodyssey/cybertron/pkg/tasks/questionanswering"
	bert_for_question_answering "github.com/nlpodyssey/cybertron/pkg/tasks/questionanswering/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks/text2text"
	bart_for_text_to_text "github.com/nlpodyssey/cybertron/pkg/tasks/text2text/bart"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textclassification"
	bert_for_text_classification "github.com/nlpodyssey/cybertron/pkg/tasks/textclassification/bert"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"
	bert_for_text_encoding "github.com/nlpodyssey/cybertron/pkg/tasks/textencoding/bert"
	distilbert_for_text_encoding "github.com/nlpodyssey/cybertron/pkg/tasks/textencoding/distilbert"
	"github.com/nlpodyssey/cybertron/pkg/tasks/tokenclassification"
	bert_for_token_classification "github.com/nlpodyssey/cybertron/pkg/tasks/tokenclassification/bert"
	flair_for_token_classification "github.com/nlpodyssey/cybertron/pkg/tasks/tokenclassification/flair"
	"github.com/nlpodyssey/cybertron/pkg/tasks/zeroshotclassifier"
	bart_for_zero_shot_classification "github.com/nlpodyssey/cybertron/pkg/tasks/zeroshotclassifier/bart"
)

var (
	text2textInterface           = reflect.TypeOf((*text2text.Interface)(nil)).Elem()
	zeroshotclassifierInterface  = reflect.TypeOf((*zeroshotclassifier.Interface)(nil)).Elem()
	questionansweringInterface   = reflect.TypeOf((*questionanswering.Interface)(nil)).Elem()
	textclassificationInterface  = reflect.TypeOf((*textclassification.Interface)(nil)).Elem()
	tokenclassificationInterface = reflect.TypeOf((*tokenclassification.Interface)(nil)).Elem()
	textencodingInterface        = reflect.TypeOf((*textencoding.Interface)(nil)).Elem()
	languagemodelingInterface    = reflect.TypeOf((*languagemodeling.Interface)(nil)).Elem()
)

// Load loads a model from file.
func Load[T any](conf *Config) (T, error) {
	return loader[T]{conf: *conf}.load()
}

func LoadModelForTextGeneration(conf *Config) (text2text.Interface, error) {
	return Load[text2text.Interface](conf)
}

func LoadModelForQuestionAnswering(conf *Config) (questionanswering.Interface, error) {
	return Load[questionanswering.Interface](conf)
}

func LoadModelForTextClassification(conf *Config) (textclassification.Interface, error) {
	return Load[textclassification.Interface](conf)
}

func LoadModelForZeroShotTextClassification(conf *Config) (zeroshotclassifier.Interface, error) {
	return Load[zeroshotclassifier.Interface](conf)
}

func LoadModelForTextEncoding(conf *Config) (textencoding.Interface, error) {
	return Load[textencoding.Interface](conf)
}

func LoadModelLanguageModeling(conf *Config) (languagemodeling.Interface, error) {
	return Load[languagemodeling.Interface](conf)
}

func LoadModelForTokenClassification(conf *Config) (tokenclassification.Interface, error) {
	return Load[tokenclassification.Interface](conf)
}

type loader[T any] struct {
	conf Config
}

func (l loader[T]) load() (obj T, _ error) {
	loadingFunc, err := l.resolveLoadingFunc()
	if err != nil {
		return obj, err
	}

	if l.conf.ModelName == "" {
		return obj, errors.New("model name not specified")
	}
	if err := l.download(); err != nil {
		return obj, err
	}
	if err := l.convert(); err != nil {
		return obj, err
	}

	return loadingFunc()
}

func (l loader[T]) resolveLoadingFunc() (func() (T, error), error) {
	obj, t := l.reflectType()
	switch {
	case t.Implements(text2textInterface):
		return l.resolveModelForText2Text, nil
	case t.Implements(zeroshotclassifierInterface):
		return l.resolveModelForZeroShotClassification, nil
	case t.Implements(questionansweringInterface):
		return l.resolveModelForQuestionAnswering, nil
	case t.Implements(textclassificationInterface):
		return l.resolveModelForTextClassification, nil
	case t.Implements(tokenclassificationInterface):
		return l.resolveModelForTokenClassification, nil
	case t.Implements(textencodingInterface):
		return l.resolveModelForTextEncoding, nil
	case t.Implements(languagemodelingInterface):
		return l.resolveModelForLanguageModeling, nil
	default:
		return nil, fmt.Errorf("loader: invalid type %T", obj)
	}
}

func (l loader[T]) reflectType() (obj T, t reflect.Type) {
	if any(obj) == nil {
		return obj, reflect.ValueOf(&obj).Type().Elem()
	}
	return obj, reflect.ValueOf(obj).Type()
}

func (l loader[T]) resolveModelForText2Text() (obj T, _ error) {
	modelDir := l.conf.FullModelPath()
	modelConfig, err := models.ReadCommonModelConfig(modelDir, "")
	if err != nil {
		return obj, err
	}

	switch modelConfig.ModelType {
	case "bart", "marian", "pegasus":
		return typeCheck[T](bart_for_text_to_text.LoadText2Text(modelDir))
	default:
		return obj, fmt.Errorf("model type %#v doesn't support the text generation task", modelConfig.ModelType)
	}
}

func (l loader[T]) resolveModelForZeroShotClassification() (obj T, _ error) {
	modelDir := l.conf.FullModelPath()
	modelConfig, err := models.ReadCommonModelConfig(modelDir, "")
	if err != nil {
		return obj, err
	}

	switch modelConfig.ModelType {
	case "bart":
		return typeCheck[T](bart_for_zero_shot_classification.LoadZeroShotClassifier(modelDir))
	default:
		return obj, fmt.Errorf("model type %#v doesn't support the zero-shot classification task", modelConfig.ModelType)
	}
}

func (l loader[T]) resolveModelForQuestionAnswering() (obj T, _ error) {
	modelDir := l.conf.FullModelPath()
	modelConfig, err := models.ReadCommonModelConfig(modelDir, "")
	if err != nil {
		return obj, err
	}

	switch modelConfig.ModelType {
	case "bert":
		return typeCheck[T](bert_for_question_answering.LoadQuestionAnswering(modelDir))
	default:
		return obj, fmt.Errorf("model type %#v doesn't support the question-answering task", modelConfig.ModelType)
	}
}

func (l loader[T]) resolveModelForTextClassification() (obj T, _ error) {
	modelDir := l.conf.FullModelPath()
	modelConfig, err := models.ReadCommonModelConfig(modelDir, "")
	if err != nil {
		return obj, err
	}

	switch modelConfig.ModelType {
	case "bert":
		return typeCheck[T](bert_for_text_classification.LoadTextClassification(modelDir))
	default:
		return obj, fmt.Errorf("model type %#v doesn't support the text classification task", modelConfig.ModelType)
	}
}

func (l loader[T]) resolveModelForTokenClassification() (obj T, _ error) {
	modelDir := l.conf.FullModelPath()
	modelConfig, err := models.ReadCommonModelConfig(modelDir, "")
	if err != nil {
		return obj, err
	}

	switch modelConfig.ModelType {
	case "bert":
		return typeCheck[T](bert_for_token_classification.LoadTokenClassification(modelDir))
	case "flair":
		return typeCheck[T](flair_for_token_classification.LoadTokenClassification(modelDir))
	default:
		return obj, fmt.Errorf("model type %#v doesn't support the token classification task", modelConfig.ModelType)
	}
}

func (l loader[T]) resolveModelForTextEncoding() (obj T, _ error) {
	modelDir := l.conf.FullModelPath()
	modelConfig, err := models.ReadCommonModelConfig(modelDir, "")
	if err != nil {
		return obj, err
	}

	switch modelConfig.ModelType {
	case "bert":
		return typeCheck[T](bert_for_text_encoding.LoadTextEncoding(modelDir))
	case "distilbert":
		return typeCheck[T](distilbert_for_text_encoding.LoadTextEncoding(modelDir))
	default:
		return obj, fmt.Errorf("model type %#v doesn't support the text encoding task", modelConfig.ModelType)
	}
}

func (l loader[T]) resolveModelForLanguageModeling() (obj T, _ error) {
	modelDir := l.conf.FullModelPath()
	modelConfig, err := models.ReadCommonModelConfig(modelDir, "")
	if err != nil {
		return obj, err
	}

	switch modelConfig.ModelType {
	case "bert":
		return typeCheck[T](bert_for_language_modeling.LoadMaskedLanguageModel(modelDir))
	case "distilbert":
		return typeCheck[T](distilbert_for_language_modeling.LoadMaskedLanguageModel(modelDir))
	default:
		return obj, fmt.Errorf("model type %#v doesn't support the language modeling task", modelConfig.ModelType)
	}
}

func typeCheck[T any](i any, err error) (T, error) {
	var empty T
	if err != nil {
		return empty, err
	}
	if mm, ok := i.(T); ok {
		return mm, nil
	}
	return empty, fmt.Errorf("unexpected type: %T", i)
}

func (l loader[T]) download() error {
	var overwriteIfExists bool
	switch l.conf.DownloadPolicy {
	case DownloadNever:
		return nil
	case DownloadAlways:
		overwriteIfExists = true
	case DownloadMissing:
		overwriteIfExists = false
	default:
		return fmt.Errorf("invalid model download policy: %#v", l.conf.DownloadPolicy)
	}
	return downloader.Download(l.conf.ModelsDir, l.conf.ModelName, overwriteIfExists, l.conf.HubAccessToken)
}

func (l loader[T]) convert() error {
	var overwriteIfExists bool
	switch l.conf.ConversionPolicy {
	case ConvertNever:
		return nil
	case ConvertAlways:
		overwriteIfExists = true
	case ConvertMissing:
		overwriteIfExists = false
	default:
		return fmt.Errorf("invalid model conversion policy: %#v", l.conf.ConversionPrecision)
	}

	modelPath := l.conf.FullModelPath()

	var err error
	switch l.conf.ConversionPrecision {
	case F32:
		err = converter.Convert[float32](modelPath, overwriteIfExists)
	case F64:
		err = converter.Convert[float64](modelPath, overwriteIfExists)
	default:
		return fmt.Errorf("invalid model conversion precision: %#v", l.conf.ConversionPrecision)
	}
	if err != nil {
		return fmt.Errorf("failed to convert model: %w", err)
	}
	return nil
}
