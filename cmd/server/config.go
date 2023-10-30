// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/server"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

// TaskType is the task type.
type TaskType string

const (
	TextGenerationTask         TaskType = "text-generation"
	ZeroShotClassificationTask TaskType = "zero-shot-classification"
	QuestionAnsweringTask      TaskType = "question-answering"
	TextClassificationTask     TaskType = "text-classification"
	TokenClassificationTask    TaskType = "token-classification"
	TextEncodingTask           TaskType = "text-encoding"
	LanguageModelingTask       TaskType = "language-modeling"
)

// TaskTypeValues is the list of supported task types.
var TaskTypeValues = []TaskType{
	TextGenerationTask,
	ZeroShotClassificationTask,
	QuestionAnsweringTask,
	TextClassificationTask,
	TokenClassificationTask,
	TextEncodingTask,
	LanguageModelingTask,
}

// ParseTaskType parses a task type.
func ParseTaskType(s string) (TaskType, error) {
	for _, v := range TaskTypeValues {
		if s == string(v) {
			return v, nil
		}
	}
	return "", fmt.Errorf("invalid task type value %#v", s)
}

// config represents the configuration of the server.
type config struct {
	task         TaskType
	loaderConfig *tasks.Config
	serverConfig *server.Config
}

// loadEnv loads config values from environment variables.
func (conf *config) loadEnv() error {
	if v, ok := os.LookupEnv("CYBERTRON_LOGLEVEL"); ok {
		l, err := zerolog.ParseLevel(v)
		if err != nil {
			log.Warn().Err(err).Msg("failed to parse env var CYBERTRON_LOGLEVEL")
		}
		zerolog.SetGlobalLevel(l)
	}

	mm := conf.loaderConfig
	lookupEnv("MODELS_DIR", &mm.ModelsDir)
	lookupEnv("MODEL", &mm.ModelName)
	lookupEnv("HUB_ACCESS_TOKEN", &mm.HubAccessToken)
	if err := lookupEnvAndParse("MODEL_DOWNLOAD", tasks.ParseDownloadPolicy, &mm.DownloadPolicy); err != nil {
		return err
	}
	if err := lookupEnvAndParse("MODEL_CONVERSION", tasks.ParseConversionPolicy, &mm.ConversionPolicy); err != nil {
		return err
	}
	if err := lookupEnvAndParse("MODEL_CONVERSION_PRECISION", tasks.ParseFloatPrecision, &mm.ConversionPrecision); err != nil {
		return err
	}
	if err := lookupEnvAndParse("MODEL_TASK", ParseTaskType, &conf.task); err != nil {
		return err
	}

	s := conf.serverConfig
	lookupEnv("NETWORK", &s.Network)
	lookupEnv("ADDRESS", &s.Address)
	if err := lookupEnvAndParse("ALLOWED_ORIGINS", parseCommaSplit, &s.AllowedOrigins); err != nil {
		return err
	}
	if err := lookupEnvAndParse("TLS_ENABLED", parseBool, &s.TLSEnabled); err != nil {
		return err
	}
	lookupEnv("TLS_CERT", &s.TLSCert)
	lookupEnv("TLS_KEY", &s.TLSKey)

	return nil
}

// bindFlagSet prepares the given FlagSet defining all necessary flags for
// setting config properties.
//
// The flags are defined using FlagSet.Func, so that if a command line flag is
// not encountered, its related config value is not overridden with any default.
func (conf *config) bindFlagSet(fs *flag.FlagSet) {
	fs.Func("loglevel", "zerolog global level", func(v string) error {
		l, err := zerolog.ParseLevel(v)
		if err != nil {
			log.Warn().Err(err).Msg("failed to parse -loglevel flag")
		}
		zerolog.SetGlobalLevel(l)
		return nil
	})

	mm := conf.loaderConfig
	fs.Func("models-dir", "models's base directory", flagAssignFunc(&mm.ModelsDir))
	fs.Func("model", "model name (and sub-path of models-dir)", flagAssignFunc(&mm.ModelName))
	fs.Func("hub-access-token", `access token to download private models from the Hugging Face Hub (optional)`, flagAssignFunc(&mm.HubAccessToken))
	fs.Func("model-download", `model downloading policy ("always"|"missing"|"never")`,
		flagParseFunc(tasks.ParseDownloadPolicy, &mm.DownloadPolicy))
	fs.Func("model-conversion", `model conversion policy ("always"|"missing"|"never")`,
		flagParseFunc(tasks.ParseConversionPolicy, &mm.ConversionPolicy))
	fs.Func("model-conversion-precision", `floating-point bits of precision to use if the model is converted ("32"|"64")`,
		flagParseFunc(tasks.ParseFloatPrecision, &mm.ConversionPrecision))
	fs.Func("task", `type of inference/computation that the model can fulfill ("text-generation"|"zero-shot-classification"|"question-answering"|"text-classification"|"token-classification"|"text-encoding"|"language-modeling")`,
		flagParseFunc(ParseTaskType, &conf.task))

	s := conf.serverConfig
	fs.Func("network", "network type for server listening", flagAssignFunc(&s.Network))
	fs.Func("address", "server listening address", flagAssignFunc(&s.Address))
	fs.Func("allowed-origins", `allowed origins (comma separated)`,
		flagParseFunc(parseCommaSplit, &s.AllowedOrigins))
	fs.Func("tls", `whether to enable TLS ("true"|"false")`,
		flagParseFunc(parseBool, &s.TLSEnabled))
	fs.Func("tls-cert", "TLS cert filename", flagAssignFunc(&s.TLSCert))
	fs.Func("tls-key", "TLS key filename", flagAssignFunc(&s.TLSKey))
}

// lookupEnv looks up the value of the given environment variable and assign it to dest.
func lookupEnv(keySuffix string, dest *string) {
	key := fmt.Sprintf("CYBERTRON_%s", keySuffix)
	if v, ok := os.LookupEnv(key); ok {
		*dest = v
	}
}

// lookupEnvAndParse looks up the value of the given environment variable and parse it.
func lookupEnvAndParse[T any](keySuffix string, parse func(string) (T, error), dest *T) error {
	key := fmt.Sprintf("CYBERTRON_%s", keySuffix)
	if v, ok := os.LookupEnv(key); ok {
		p, err := parse(v)
		if err != nil {
			return fmt.Errorf("failed to parse env var %s: %w", key, err)
		}
		*dest = p
	}
	return nil
}

// flagAssignFunc returns a function that assigns the given value to the given dest.
func flagAssignFunc(dest *string) func(string) error {
	return func(v string) error {
		*dest = v
		return nil
	}
}

// flagParseFunc returns a function that parses the given value and assigns it to the given dest.
func flagParseFunc[T any](parse func(string) (T, error), dest *T) func(string) error {
	return func(v string) error {
		p, err := parse(v)
		if err != nil {
			return err
		}
		*dest = p
		return nil
	}
}

// parseCommaSplit parses the given string as a comma-separated list of strings.
func parseCommaSplit(s string) ([]string, error) {
	return strings.Split(s, ","), nil
}

// parseBool parses the given string as a boolean.
func parseBool(s string) (bool, error) {
	switch s {
	case "true":
		return true, nil
	case "false":
		return false, nil
	default:
		return false, fmt.Errorf("invalid boolean value %#v", s)
	}
}
