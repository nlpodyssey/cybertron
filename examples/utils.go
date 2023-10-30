// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package examples

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/joho/godotenv"
	"github.com/rs/zerolog/log"
)

// ForEachInput calls the given callback function for each line of input.
func ForEachInput(r io.Reader, callback func(text string) error) error {
	scanner := bufio.NewScanner(r)
	for {
		fmt.Print("> ")
		scanner.Scan()
		text := scanner.Text()
		if text == "" {
			break
		}
		if err := callback(text); err != nil {
			return err
		}
	}
	return nil
}

// HasEnvVar returns the value of the environment variable with the given key.
// It panics if the environment variable is not set.
func HasEnvVar(key string) string {
	value := os.Getenv(key)
	if value == "" || len(strings.Trim(value, " ")) == 0 {
		log.Fatal().Msgf("missing env var: %s", key)
	}
	return value
}

// HasEnvVarOr returns the value of the environment variable with the given key.
// It returns the alternative value if the environment variable is not set.
func HasEnvVarOr(key string, alt string) string {
	value := os.Getenv(key)
	if value == "" || len(strings.Trim(value, " ")) == 0 {
		return alt
	}
	return value
}

// MarshalJSON returns the JSON string representation of the input data
func MarshalJSON(data any) string {
	m, _ := json.MarshalIndent(data, "", "  ")
	return string(m)
}

// LoadDotenv loads the .env file if it exists.
func LoadDotenv() {
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
