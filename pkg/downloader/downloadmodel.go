// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package downloader

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/nlpodyssey/cybertron/pkg/models"
	"github.com/rs/zerolog/log"
)

const (
	// Hugging Face repository URL, in the format:
	// "https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
	huggingFaceCoPrefix = "https://huggingface.co/%s/resolve/%s/%s"
	// Default revision name for fetching model from Hugging Face repository
	defaultRevision = "main"
)

// supportedModelsFiles contains the set of all supported model types as keys,
// mapped with the set of all related files to download.
var supportedModelsFiles = map[string][]string{
	"bart":    {"pytorch_model.bin", "vocab.json", "merges.txt"},
	"pegasus": {"pytorch_model.bin", "spiece.model"},
	"marian":  {"pytorch_model.bin", "vocab.json", "source.spm", "target.spm"},
	"bert":    {"pytorch_model.bin", "vocab.txt"},
	"electra": {"pytorch_model.bin", "vocab.txt"},
}

// Download downloads a supported pre-trained model from huggingface.co
// repositories.
//
// A model typically consists of a set of different files to be downloaded.
// These files will be placed in the full model's path, which corresponds to
// the joining of the targetPath and the modelName. This usually ends up
// in one or more additional directories nested under the targetPath, depending
// on the modelName.
//
// If one or more directory levels don't yet exist, they are created
// setting the permissions bits to 0755 (rwxr-xr-x).
//
// By setting the flag overwriteIfExist to false, any file that already
// exists is kept and considered as already successfully downloaded. If
// the flag is otherwise set to true, existing files will be forcefully
// downloaded and overwritten.
func Download(modelsDir, modelName string, overwriteIfExists bool) error {
	return downloader{
		modelPath:        filepath.Join(modelsDir, modelName),
		modelName:        modelName,
		overwriteIfExist: overwriteIfExists,
	}.download()
}

// downloader is a helper struct for downloading a model.
type downloader struct {
	modelPath        string
	modelName        string
	overwriteIfExist bool
}

func (d downloader) download() error {
	if err := d.ensureModelPath(); err != nil {
		return err
	}
	if err := d.downloadFile(models.DefaultModelConfigFilename); err != nil {
		return err
	}
	return d.downloadModelSpecificFiles()
}

func (d downloader) ensureModelPath() error {
	if info, err := os.Stat(d.modelPath); err == nil && info.IsDir() {
		return nil
	}
	if err := os.MkdirAll(d.modelPath, 0755); err != nil {
		return fmt.Errorf("error creating model path %#v: %w", d.modelPath, err)
	}
	return nil
}

func (d downloader) downloadModelSpecificFiles() error {
	config, err := models.ReadCommonModelConfig(d.modelPath, "")
	if err != nil {
		return err
	}

	filenames, isSupported := supportedModelsFiles[config.ModelType]
	if !isSupported {
		return fmt.Errorf("unsupported model type: %#v", config.ModelType)
	}

	for _, filename := range filenames {
		if err := d.downloadFile(filename); err != nil {
			return err
		}
	}
	return nil
}

func (d downloader) downloadFile(name string) (err error) {
	fPath := filepath.Join(d.modelPath, name)
	if info, err := os.Stat(fPath); !d.overwriteIfExist && err == nil && !info.IsDir() {
		log.Debug().Str("file", fPath).Msg("model file already exists, skipping download")
		return nil
	}

	url := d.bucketURL(name)
	log.Debug().Str("url", url).Str("destination", fPath).Msg("downloading")

	f, err := os.Create(fPath)
	if err != nil {
		return fmt.Errorf("error creating file %#v: %w", fPath, err)
	}
	defer func() {
		if e := f.Close(); e != nil && err == nil {
			err = fmt.Errorf("error closing file %#v: %w", fPath, e)
		}
	}()

	resp, err := http.Get(url)
	if err != nil {
		return fmt.Errorf("error getting %#v: %w", url, err)
	}
	defer func() {
		if e := resp.Body.Close(); e != nil && err == nil {
			err = fmt.Errorf("error closing %#v response body: %w", url, e)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%#v responded with %s", url, resp.Status)
	}

	prog := newDownloadProgress(int(resp.ContentLength))
	prog.Start()
	defer prog.Stop()

	_, err = io.Copy(f, io.TeeReader(resp.Body, prog))
	if err != nil {
		return fmt.Errorf("error downloading %#v to %#v: %w", url, fPath, err)
	}
	return nil
}

func (d downloader) bucketURL(fileName string) string {
	return fmt.Sprintf(huggingFaceCoPrefix, d.modelName, defaultRevision, fileName)
}
