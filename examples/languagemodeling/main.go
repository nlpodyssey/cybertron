// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"fmt"
	"os"

	//lint:ignore ST1001 allow dot import just to make the example more readable
	. "github.com/nlpodyssey/cybertron/examples"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/languagemodeling"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
	LoadDotenv()

	modelsDir := HasEnvVarOr("CYBERTRON_MODELS_DIR", "models")
	modelName := HasEnvVarOr("CYBERTRON_MODEL", languagemodeling.DefaultModel)

	m, err := tasks.Load[languagemodeling.Interface](&tasks.Config{ModelsDir: modelsDir, ModelName: modelName})
	if err != nil {
		log.Fatal().Err(err).Send()
	}

	fn := func(text string) error {
		result, err := m.Predict(context.Background(), text, languagemodeling.Parameters{K: 10})
		if err != nil {
			return err
		}
		fmt.Println(MarshalJSON(result))
		return nil
	}

	err = ForEachInput(os.Stdin, fn)
	if err != nil {
		log.Fatal().Err(err).Send()
	}
}
