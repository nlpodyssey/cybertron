// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"errors"
	"flag"
	"io"
	"os"
	"os/signal"
	"time"

	"github.com/joho/godotenv"
	"github.com/nlpodyssey/cybertron/pkg/server"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/text2text"
	"github.com/nlpodyssey/cybertron/pkg/tasks/zeroshotclassifier"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

const defaultModelsDir = "models"

// main is the entry point of the application.
func main() {
	if err := run(); err != nil {
		log.Error().Err(err).Send()
		os.Exit(1)
	}
}

// run set the configuration and starts the server.
func run() error {
	initLogger()
	loadDotenv()

	conf := &config{
		loaderConfig: tasks.DefaultConfig(defaultModelsDir, ""),
		serverConfig: server.DefaultServerConfig(),
	}

	// load env vars values *before* parsing command line flags:
	// this gives to the flag a priority over values from the environment.
	if err := conf.loadEnv(); err != nil {
		return err
	}

	fs := flag.NewFlagSet(os.Args[0], flag.ContinueOnError)
	conf.bindFlagSet(fs)

	err := fs.Parse(os.Args[1:])
	if errors.Is(err, flag.ErrHelp) {
		return nil
	}
	if err != nil {
		return err
	}

	var m any
	switch conf.task {
	case ZeroShotClassificationTask:
		m, err = tasks.Load[zeroshotclassifier.Interface](conf.loaderConfig)
	case Text2TextTask:
		m, err = tasks.Load[text2text.Interface](conf.loaderConfig)
	}
	if err != nil {
		return err
	}
	if i, ok := m.(io.Closer); ok {
		defer i.Close()
	}

	s, err := server.New(conf.serverConfig, m)
	if err != nil {
		return err
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, os.Kill)
	defer stop()

	return s.Run(ctx)
}

// initLogger initializes the logger.
func initLogger() {
	log.Logger = log.Output(zerolog.ConsoleWriter{
		Out:        os.Stderr,
		TimeFormat: time.RFC3339,
	})
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
}

// loadDotenv loads the .env file if it exists.
func loadDotenv() {
	_, err := os.Stat(".env")
	if os.IsNotExist(err) {
		return
	}
	if err != nil {
		log.Warn().Err(err).Msg("failed to read .env file")
		return
	}
	err = godotenv.Load()
	if err != nil {
		log.Warn().Err(err).Msg("failed to read .env file")
	}
}
