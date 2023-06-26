// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package converter

import (
	"fmt"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/converter/bart"
	"github.com/nlpodyssey/cybertron/pkg/converter/bert"
	"github.com/nlpodyssey/cybertron/pkg/converter/distilbert"
	"github.com/nlpodyssey/cybertron/pkg/converter/flair"
	"github.com/nlpodyssey/cybertron/pkg/models"
	"github.com/nlpodyssey/spago/mat/float"
)

// Convert automatically converts a supported pre-trained model, already
// downloaded from huggingface.co repositories, to a format usable by Spago machine learning framework.
//
// It accepts the path to the model's directory and creates the converted
// files in the same place.
func Convert[T float.DType](modelPath string, overwriteIfExists bool) error {
	modelType, err := resolveModelType(modelPath)
	if err != nil {
		return err
	}

	switch modelType {
	case "bert", "electra":
		return bert.Convert[T](modelPath, overwriteIfExists)
	case "distilbert":
		return distilbert.Convert[T](modelPath, overwriteIfExists)
	case "bart", "marian", "pegasus":
		return bart.Convert[T](modelPath, overwriteIfExists)
	case "flair":
		return flair.Convert[T](modelPath, overwriteIfExists)
	default:
		return fmt.Errorf("unsupported model type: %#v", modelType)
	}
}

func resolveModelType(modelPath string) (string, error) {
	if strings.Contains(modelPath, "flair") {
		// Handling the case where there is no configuration file
		return "flair", nil
	}

	config, err := models.ReadCommonModelConfig(modelPath, "")
	if err != nil {
		return "", err
	}
	return config.ModelType, nil
}
