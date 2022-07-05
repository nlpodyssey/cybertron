// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"
	"time"

	. "github.com/nlpodyssey/cybertron/examples"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/questionanswering"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	zerolog.SetGlobalLevel(zerolog.TraceLevel)

	modelsDir := HasEnvVar("CYBERTRON_MODELS_DIR")
	modelName := HasEnvVar("CYBERTRON_MODEL")
	paragraph := HasEnvVar("CYBERTRON_QA_PARAGRAPH")

	m, err := tasks.Load[questionanswering.Interface](&tasks.Config{ModelsDir: modelsDir, ModelName: modelName})
	if err != nil {
		log.Fatal().Err(err).Send()
	}
	defer tasks.Finalize(m)

	opts := questionanswering.Options{
		MaxAnswerLength: 0,
		MinScore:        0,
	}

	fn := func(text string) error {
		start := time.Now()
		result, err := m.Answer(text, paragraph, opts)
		if err != nil {
			return err
		}
		fmt.Println(time.Since(start).Seconds())
		fmt.Printf("%+v\n", result)
		return nil
	}

	fmt.Println(paragraph)

	err = ForEachInput(os.Stdin, fn)
	if err != nil {
		log.Fatal().Err(err)
	}
}
