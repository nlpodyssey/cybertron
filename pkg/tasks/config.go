// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tasks

import (
	"fmt"
	"path/filepath"
)

// DownloadPolicy is a policy for downloading a model.
type DownloadPolicy string

// ConversionPolicy is a policy for converting a pre-trained model.
type ConversionPolicy string

// FloatPrecision is the floating-point precision of the converted model.
type FloatPrecision string

const (
	// DownloadAlways means that the model will be downloaded even if it already exists.
	DownloadAlways DownloadPolicy = "always"
	// DownloadMissing means that the model will be downloaded only if it doesn't exist.
	DownloadMissing DownloadPolicy = "missing"
	// DownloadNever means that the model will not be downloaded.
	DownloadNever DownloadPolicy = "never"

	// ConvertAlways converts the model even if it already exists.
	ConvertAlways ConversionPolicy = "always"
	// ConvertMissing converts the model only if it does not exist.
	ConvertMissing ConversionPolicy = "missing"
	// ConvertNever does not convert the model.
	ConvertNever ConversionPolicy = "never"

	// F32 is the 32 floating-point precision.
	F32 FloatPrecision = "32"
	// F64 is the 64 floating-point precision.
	F64 FloatPrecision = "64"
)

// Config is the configuration for the loader.
type Config struct {
	ModelsDir           string
	ModelName           string
	HubAccessToken      string
	DownloadPolicy      DownloadPolicy
	ConversionPolicy    ConversionPolicy
	ConversionPrecision FloatPrecision
}

// DefaultConfig returns the default configuration.
func DefaultConfig(modelsDir, modelName string) Config {
	return Config{
		ModelsDir:           modelsDir,
		ModelName:           modelName,
		HubAccessToken:      "",
		DownloadPolicy:      DownloadMissing,
		ConversionPolicy:    ConvertMissing,
		ConversionPrecision: F32,
	}
}

// WithHubAccessToken sets the HubAccessToken.
func (c Config) WithHubAccessToken(token string) Config {
	c.HubAccessToken = token
	return c
}

// FullModelPath returns the full model path.
func (c Config) FullModelPath() string {
	return filepath.Join(c.ModelsDir, c.ModelName)
}

// DownloadPolicyValues is a list of supported download policies.
var DownloadPolicyValues = []DownloadPolicy{DownloadAlways, DownloadMissing, DownloadNever}

// ConversionPolicyValues is a list of supported conversion policies.
var ConversionPolicyValues = []ConversionPolicy{ConvertAlways, ConvertMissing, ConvertNever}

// floatPrecisionValues is a list of supported floating-point precisions.
var floatPrecisionValues = []FloatPrecision{F32, F64}

// ParseDownloadPolicy parses a string into a download policy.
func ParseDownloadPolicy(s string) (DownloadPolicy, error) {
	for _, v := range DownloadPolicyValues {
		if s == string(v) {
			return v, nil
		}
	}
	return "", fmt.Errorf("invalid download policy value %#v", s)
}

// ParseConversionPolicy parses a string into a conversion policy.
func ParseConversionPolicy(s string) (ConversionPolicy, error) {
	for _, v := range ConversionPolicyValues {
		if s == string(v) {
			return v, nil
		}
	}
	return "", fmt.Errorf("invalid model pre-load policy value %#v", s)
}

// ParseFloatPrecision parses a string into a FloatPrecision precision type.
func ParseFloatPrecision(s string) (FloatPrecision, error) {
	for _, v := range floatPrecisionValues {
		if s == string(v) {
			return v, nil
		}
	}
	return "", fmt.Errorf("invalid model floating-point precision value %#v", s)
}
