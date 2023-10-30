// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"fmt"
	"time"

	//lint:ignore ST1001 allow dot import just to make the example more readable
	. "github.com/nlpodyssey/cybertron/examples"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textgeneration"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textgeneration/bart"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

var (
	query = "Why does water heated to room temperature feel colder than the air around it?"

	passages = []string{"when the skin is completely wet. The body continuously loses water by...",
		"at greater pressures. There is an ambiguity, however, as to the meaning of the terms 'heating' and 'cooling'...",
		"are not in a relation of thermal equilibrium, heat will flow from the hotter to the colder, by whatever pathway...",
		"air condition and moving along a line of constant enthalpy toward a state of higher humidity. A simple example ...",
		"Thermal contact conductance In physics, thermal contact conductance is the study of heat conduction between solid ...",
	}
)

func main() {
	zerolog.SetGlobalLevel(zerolog.TraceLevel)
	LoadDotenv()

	modelsDir := HasEnvVar("CYBERTRON_MODELS_DIR")

	m, err := tasks.Load[*bart.TextGeneration](&tasks.Config{
		ModelsDir: modelsDir,
		ModelName: textgeneration.DefaultModelForAbstractiveQuestionAnswering,
	})
	if err != nil {
		log.Fatal().Err(err).Send()
	}
	defer tasks.Finalize(m)

	opts := textgeneration.DefaultOptions()

	start := time.Now()
	result, err := m.Generate(context.Background(), textgeneration.PrepareInputForAbstractiveQuestionAnswering(query, passages), opts)
	if err != nil {
		panic(err)
	}
	fmt.Println(time.Since(start).Seconds())

	fmt.Println("> " + query)
	fmt.Println(result.Texts[0])
}
