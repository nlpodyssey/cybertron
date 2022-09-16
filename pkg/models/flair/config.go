// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"encoding/json"
	"os"
)

// Config is the configuration for the Flair architecture.
type Config struct {
	ModelType string            `json:"model_type"`
	ID2Label  map[string]string `json:"id2label"`
}

// ConfigFromFile loads a Flair model Config from file.
func ConfigFromFile(file string) (Config, error) {
	var config Config
	configFile, err := os.Open(file)
	if err != nil {
		return Config{}, err
	}
	defer configFile.Close()
	err = json.NewDecoder(configFile).Decode(&config)
	if err != nil {
		return Config{}, err
	}
	return config, nil
}
