// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package converter

import (
	"fmt"
	"github.com/nlpodyssey/cybertron/pkg/converter/bert"

	"github.com/nlpodyssey/cybertron/pkg/converter/bart"
	"github.com/nlpodyssey/cybertron/pkg/models"
	"github.com/nlpodyssey/spago/mat/float"
)

// Convert automatically converts a supported pre-trained model, already
// downloaded from huggingface.co repositories, to a format usable by Spago machine learning framework.
//
// It accepts the path to the model's directory and creates the converted
// files in the same place.
func Convert[T float.DType](modelPath string, overwriteIfExists bool) error {
	config, err := models.ReadCommonModelConfig(modelPath, "")
	if err != nil {
		return err
	}
	switch config.ModelType {
	case "bert", "electra":
		return bert.Convert[T](modelPath, overwriteIfExists)
	case "bart", "marian", "pegasus":
		return bart.Convert[T](modelPath, overwriteIfExists)
	default:
		return fmt.Errorf("unsupported model type: %#v", config.ModelType)
	}
}
