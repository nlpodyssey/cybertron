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
	"github.com/nlpodyssey/cybertron/pkg/tasks/questionanswering"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// Example of content to be used as context for the question answering task.
const content = `Cloud computing is a technology that allows individuals and businesses to access computing resources over the Internet. It enables users to utilize hardware and software that are managed by third parties at remote locations. Services provided by cloud computing include storage solutions, databases, and computing power, which can be used on a pay-per-use basis. This model offers flexibility and scalability, reducing the need for large upfront investments in infrastructure. Major providers of cloud computing services include Amazon Web Services (AWS), Microsoft Azure, and Google Cloud Platform (GCP).`

func main() {
	zerolog.SetGlobalLevel(zerolog.TraceLevel)
	LoadDotenv()

	modelsDir := HasEnvVarOr("CYBERTRON_MODELS_DIR", "models")
	modelName := HasEnvVarOr("CYBERTRON_MODEL", questionanswering.DefaultEnglishModel)

	m, err := tasks.Load[questionanswering.Interface](&tasks.Config{ModelsDir: modelsDir, ModelName: modelName})
	if err != nil {
		log.Fatal().Err(err).Send()
	}

	opts := &questionanswering.Options{}

	fn := func(text string) error {
		start := time.Now()
		result, err := m.ExtractAnswer(context.Background(), text, content, opts)
		if err != nil {
			return err
		}
		fmt.Println(time.Since(start).Seconds())
		fmt.Println(MarshalJSON(result))
		return nil
	}

	fmt.Println(content)

	err = ForEachInput(os.Stdin, fn)
	if err != nil {
		log.Fatal().Err(err).Send()
	}
}
