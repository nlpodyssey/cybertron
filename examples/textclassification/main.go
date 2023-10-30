// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"fmt"
	"os"
	"time"

	//lint:ignore ST1001 allow dot import just to make the example more readable
	. "github.com/nlpodyssey/cybertron/examples"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textclassification"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

const limit = 5 // number of labels to show

func main() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
	LoadDotenv()

	modelsDir := HasEnvVarOr("CYBERTRON_MODELS_DIR", "models")
	modelName := HasEnvVarOr("CYBERTRON_MODEL", textclassification.DefaultModelForGeographicCategorizationMulti)

	m, err := tasks.Load[textclassification.Interface](&tasks.Config{ModelsDir: modelsDir, ModelName: modelName})
	if err != nil {
		log.Fatal().Err(err).Send()
	}

	fn := func(text string) error {
		start := time.Now()
		result, err := m.Classify(context.Background(), text)
		if err != nil {
			return err
		}
		fmt.Println(time.Since(start).Seconds())

		for i := range result.Labels[:limit] {
			fmt.Printf("%s\t%0.3f\n", result.Labels[i], result.Scores[i])
		}
		return nil
	}

	err = ForEachInput(os.Stdin, fn)
	if err != nil {
		log.Fatal().Err(err).Send()
	}
}
