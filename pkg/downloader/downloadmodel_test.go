// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package downloader

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDownloadModel(t *testing.T) {
	if os.Getenv("TEST_MODEL_DOWNLOAD") == "" {
		// We don't want to flood Hugging Face with real requests every the tests run.
		t.Skip("skipping test - set env var TEST_MODEL_DOWNLOAD to run this test")
	}
	dir := t.TempDir()
	modelName := "hf-internal-testing/tiny-random-bart"

	err := Download(dir, modelName, false)

	require.NoError(t, err)

	assert.DirExists(t, filepath.Join(dir, modelName))
	assert.FileExists(t, filepath.Join(dir, modelName, "config.json"))
	assert.FileExists(t, filepath.Join(dir, modelName, "pytorch_model.bin"))
	assert.FileExists(t, filepath.Join(dir, modelName, "vocab.json"))
	assert.FileExists(t, filepath.Join(dir, modelName, "merges.txt"))
}
