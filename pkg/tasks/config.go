// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tasks

import (
	"fmt"
	"path/filepath"
)

// DownloadPolicy is a policy for downloading a model.
type DownloadPolicy int

// ConversionPolicy is a policy for converting a pre-trained model.
type ConversionPolicy int

// FloatPrecision is the floating-point precision of the converted model.
type FloatPrecision int

const (
	// DownloadMissing means that the model will be downloaded only if it doesn't exist.
	DownloadMissing DownloadPolicy = iota
	// DownloadAlways means that the model will be downloaded even if it already exists.
	DownloadAlways
	// DownloadNever means that the model will not be downloaded.
	DownloadNever
)

const (
	// ConvertMissing converts the model only if it does not exist.
	ConvertMissing ConversionPolicy = iota
	// ConvertAlways converts the model even if it already exists.
	ConvertAlways
	// ConvertNever does not convert the model.
	ConvertNever
)

const (
	// F32 is the 32 floating-point precision.
	F32 FloatPrecision = iota
	// F64 is the 64 floating-point precision.
	F64
)

// Config is the configuration for the loader.
type Config struct {
	// ModelsDir is the directory where the models are stored.
	ModelsDir string
	// ModelName is the name of the model (format: <org>/<model>).
	ModelName string
	// HubAccessToken is the access token for the Hugging Face Hub.
	HubAccessToken string
	// DownloadPolicy is the policy for downloading the model (default missing)
	DownloadPolicy DownloadPolicy
	// ConversionPolicy is the policy for converting the model (default missing)
	ConversionPolicy ConversionPolicy
	// ConversionPrecision is the floating-point precision of the converted model (default 32)
	ConversionPrecision FloatPrecision
}

// FullModelPath returns the full model path.
func (c *Config) FullModelPath() string {
	return filepath.Join(c.ModelsDir, c.ModelName)
}

// downloadPolicyValues is a list of supported download policies.
var downloadPolicyValues = map[string]DownloadPolicy{
	"missing": DownloadMissing,
	"always":  DownloadAlways,
	"never":   DownloadNever,
}

// conversionPolicyValues is a list of supported conversion policies.
var conversionPolicyValues = map[string]ConversionPolicy{
	"missing": ConvertMissing,
	"always":  ConvertAlways,
	"never":   ConvertNever,
}

// floatPrecisionValues is a list of supported floating-point precisions.
var floatPrecisionValues = map[string]FloatPrecision{
	"32": F32,
	"64": F64,
}

// ParseDownloadPolicy parses a string into a download policy.
func ParseDownloadPolicy(s string) (DownloadPolicy, error) {
	result, ok := downloadPolicyValues[s]
	if !ok {
		return 0, fmt.Errorf("invalid model download policy value %#v", s)
	}
	return result, nil
}

// ParseConversionPolicy parses a string into a conversion policy.
func ParseConversionPolicy(s string) (ConversionPolicy, error) {
	result, ok := conversionPolicyValues[s]
	if !ok {
		return 0, fmt.Errorf("invalid model conversion policy value %#v", s)
	}
	return result, nil
}

// ParseFloatPrecision parses a string into a FloatPrecision precision type.
func ParseFloatPrecision(s string) (FloatPrecision, error) {
	result, ok := floatPrecisionValues[s]
	if !ok {
		return 0, fmt.Errorf("invalid model floating-point precision value %#v", s)
	}
	return result, nil
}
