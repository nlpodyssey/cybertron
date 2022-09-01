// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

// Config provides configuration settings for a Character-level Language Model.
type Config struct {
	Name              string
	VocabularySize    int
	EmbeddingSize     int
	HiddenSize        int
	OutputSize        int // use the projection layer when the output size is > 0
	SequenceSeparator string
	UnknownToken      string
	Trainable         bool
}
