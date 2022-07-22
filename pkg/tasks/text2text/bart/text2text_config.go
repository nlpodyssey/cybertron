// Copyright 2021 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bart

import (
	"github.com/nlpodyssey/cybertron/pkg/generationutils"
	"github.com/nlpodyssey/cybertron/pkg/models/bart"
)

// decoderConfig converts the Bart model Config to a generationutils.Config.
func decoderConfig(c bart.Config) generationutils.Config {
	lengthPenalty := c.LengthPenalty
	if lengthPenalty == 0 {
		lengthPenalty = 1.0
	}
	return generationutils.Config{
		NumBeams:            c.NumBeams,
		MinLength:           0,
		MaxLength:           c.MaxLength,
		IsEncoderDecoder:    c.IsEncoderDecoder,
		BOSTokenID:          c.BosTokenID,
		EOSTokenID:          c.EosTokenID,
		PadTokenID:          c.PadTokenID,
		VocabSize:           c.VocabSize,
		DecoderStartTokenID: c.DecoderStartTokenID,
		LengthPenalty:       lengthPenalty,
		EarlyStopping:       false,
		BadWordsIDs:         c.BadWordsIDs,
	}
}
