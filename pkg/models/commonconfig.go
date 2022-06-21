// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package models

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// DefaultModelConfigFilename is the default configuration filename for all
// compatible pre-trained model from Hugging Face.
const DefaultModelConfigFilename = "config.json"

// CommonModelConfig provides the bare minimum set of model configuration
// properties which are shared among model of different types.
//
// This is useful when you need to perform different actions depending on
// the value of certain basic common settings.
//
// For example, the operations of downloading and converting Hugging Face models
// use this information to roughly validate the JSON configuration data, to
// read the specific type of model, and then to decide if and how to proceed
// with further actions (downloading further files, converting the model, ...).
type CommonModelConfig struct {
	ModelType string `json:"model_type"`
}

// ReadCommonModelConfig parses the main JSON configuration file of a Hugging
// Face's model, returning a new CommonModelConfig value.
//
// The function accepts the path to the model's directory and the optional
// name of the configuration file. If configFilename is an empty string,
// the value of DefaultModelConfigFilename is used instead.
func ReadCommonModelConfig(modelPath, configFilename string) (conf *CommonModelConfig, err error) {
	if configFilename == "" {
		configFilename = DefaultModelConfigFilename
	}
	name := filepath.Join(modelPath, configFilename)
	f, err := os.Open(name)
	if err != nil {
		return nil, fmt.Errorf("error opening config file %#v: %w", name, err)
	}
	defer func() {
		if e := f.Close(); e != nil && err == nil {
			err = fmt.Errorf("error closing config file %#v: %w", name, e)
		}
	}()

	err = json.NewDecoder(f).Decode(&conf)
	if err != nil {
		return nil, fmt.Errorf("error parsing JSON config file %#v: %w", name, err)
	}
	return conf, nil
}
