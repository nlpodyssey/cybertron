// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package models

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReadCommonModelConfig(t *testing.T) {
	const validConfigJSON = `{ "model_type": "foo", "other_stuff": "..." }`

	var validConfig = &CommonModelConfig{ModelType: "foo"}

	t.Run("correct parsing of config file with default name", func(t *testing.T) {
		dir := t.TempDir()
		createConfigFile(t, filepath.Join(dir, "config.json"), validConfigJSON)
		conf, err := ReadCommonModelConfig(dir, "")
		assert.NoError(t, err)
		assert.Equal(t, validConfig, conf)
	})

	t.Run("correct parsing of config file with custom name", func(t *testing.T) {
		dir := t.TempDir()
		createConfigFile(t, filepath.Join(dir, "foo.bar"), validConfigJSON)
		conf, err := ReadCommonModelConfig(dir, "foo.bar")
		assert.NoError(t, err)
		assert.Equal(t, validConfig, conf)
	})

	t.Run("missing model type", func(t *testing.T) {
		dir := t.TempDir()
		createConfigFile(t, filepath.Join(dir, "config.json"), "{}")
		conf, err := ReadCommonModelConfig(dir, "")
		assert.NoError(t, err)
		expected := &CommonModelConfig{ModelType: ""}
		assert.Equal(t, expected, conf)
	})

	t.Run("file not found", func(t *testing.T) {
		dir := t.TempDir()
		conf, err := ReadCommonModelConfig(dir, "")
		assert.Error(t, err)
		assert.Nil(t, conf)
	})

	t.Run("empty file", func(t *testing.T) {
		dir := t.TempDir()
		createConfigFile(t, filepath.Join(dir, "config.json"), "")
		conf, err := ReadCommonModelConfig(dir, "")
		assert.Error(t, err)
		assert.Nil(t, conf)
	})

	t.Run("invalid JSON", func(t *testing.T) {
		dir := t.TempDir()
		createConfigFile(t, filepath.Join(dir, "config.json"), "")
		conf, err := ReadCommonModelConfig(dir, "this is not JSON!")
		assert.Error(t, err)
		assert.Nil(t, conf)
	})
}

func createConfigFile(t *testing.T, path, content string) {
	f, err := os.Create(path)
	require.NoError(t, err)
	_, err = f.Write([]byte(content))
	require.NoError(t, err)
	require.NoError(t, f.Close())
}
