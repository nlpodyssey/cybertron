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
	"github.com/nlpodyssey/cybertron/pkg/tasks/textgeneration"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// DefaultModel is REBEL, a text generation model that performs end-to-end relation extraction
// for more than 200 different relation types.
// Model card: https://huggingface.co/Babelscape/rebel-large
const DefaultModel = "Babelscape/rebel-large"

func main() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
	LoadDotenv()

	modelsDir := HasEnvVarOr("CYBERTRON_MODELS_DIR", "models")

	m, err := tasks.LoadModelForTextGeneration(&tasks.Config{
		ModelsDir: modelsDir,
		ModelName: DefaultModel,
	})
	if err != nil {
		log.Fatal().Err(err).Send()
	}
	defer tasks.Finalize(m)

	opts := textgeneration.DefaultOptions()

	fn := func(text string) error {
		start := time.Now()
		result, err := m.Generate(context.Background(), text, opts)
		if err != nil {
			return err
		}
		fmt.Println(time.Since(start).Seconds())
		fmt.Println(MarshalJSON(ExtractTriplets(result.Texts[0])))
		return nil
	}

	err = ForEachInput(os.Stdin, fn)
	if err != nil {
		log.Fatal().Err(err).Send()
	}
}
