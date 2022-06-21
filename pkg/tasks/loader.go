// Copyright 2022 NLP Odyssey Authors. All rights reserved.
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
	"github.com/nlpodyssey/cybertron/pkg/tasks/implementations/bart"
	"github.com/nlpodyssey/cybertron/pkg/tasks/text2text"
	"github.com/nlpodyssey/cybertron/pkg/tasks/zeroshotclassifier"
)

// Load loads a model from file.
func Load[T any](conf Config) (T, error) {
	if err := checkInterface[T](); err != nil {
		var obj T
		return obj, err
	}
	l := &loader[T]{conf: conf}
	return l.load()
}

func checkInterface[T any]() error {
	var obj T
	switch i := resolveNullInterface(obj).(type) {
	case *text2text.Interface, text2text.Interface, *zeroshotclassifier.Interface, zeroshotclassifier.Interface:
		return nil
	default:
		return fmt.Errorf("loader: invalid type %T", i)
	}
}

func resolveNullInterface[T any](obj T) any {
	if any(obj) == nil {
		return reflect.ValueOf(&obj).Interface()
	}
	return obj
}

type loader[T any] struct {
	conf Config
}

func (l *loader[T]) load() (T, error) {
	var obj T

	if l.conf.ModelName == "" {
		return obj, errors.New("model name not specified")
	}
	if err := l.download(); err != nil {
		return obj, err
	}
	if err := l.convert(); err != nil {
		return obj, err
	}

	switch resolveNullInterface(obj).(type) {
	case *text2text.Interface, text2text.Interface:
		return l.resolveModelForText2Text()
	case *zeroshotclassifier.Interface, zeroshotclassifier.Interface:
		return l.resolveModelForZeroShotClassification()
	default:
		return obj, fmt.Errorf("unknown task for: %T", obj)
	}
}

func (l *loader[T]) download() error {
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
	return downloader.Download(l.conf.ModelsDir, l.conf.ModelName, overwriteIfExists)
}

func (l *loader[T]) convert() error {
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

func (l *loader[T]) resolveModelForText2Text() (T, error) {
	var obj T

	modelDir := l.conf.FullModelPath()
	modelConfig, err := models.ReadCommonModelConfig(modelDir, "")
	if err != nil {
		return obj, err
	}

	switch modelConfig.ModelType {
	case "bart", "marian", "pegasus":
		return typeCheck[T](bart.LoadText2Text(modelDir))
	default:
		return obj, fmt.Errorf("model type %#v doesn't support the text generation task", modelConfig.ModelType)
	}
}

func (l *loader[T]) resolveModelForZeroShotClassification() (T, error) {
	var obj T

	modelDir := l.conf.FullModelPath()
	modelConfig, err := models.ReadCommonModelConfig(modelDir, "")
	if err != nil {
		return obj, err
	}

	switch modelConfig.ModelType {
	case "bart":
		return typeCheck[T](bart.LoadZeroShotClassifier(modelDir))
	default:
		return obj, fmt.Errorf("model type %#v doesn't support the zero-shot classification task", modelConfig.ModelType)
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
