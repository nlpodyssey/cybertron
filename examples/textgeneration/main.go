// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"io"
	"os"
	"time"

	. "github.com/nlpodyssey/cybertron/examples"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/text2text"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)

	modelsDir := HasEnvVar("CYBERTRON_MODELS_DIR")
	modelName := HasEnvVar("CYBERTRON_MODEL")

	m, err := tasks.Load[text2text.Interface](tasks.DefaultConfig(modelsDir, modelName))
	if err != nil {
		log.Fatal().Err(err)
	}
	if m, ok := m.(io.Closer); ok {
		defer m.Close()
	}

	opts := text2text.DefaultOptions()

	fn := func(text string) error {
		start := time.Now()
		result, err := m.Generate(text, opts)
		if err != nil {
			return err
		}
		fmt.Println(time.Since(start).Seconds())
		fmt.Println(result.Texts[0])
		return nil
	}

	err = ForEachInput(os.Stdin, fn)
	if err != nil {
		log.Fatal().Err(err)
	}
}
