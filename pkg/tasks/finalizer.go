// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tasks

import (
	"io"

	"github.com/rs/zerolog/log"
)

// Finalize finalizes the structures i.e. closes the underlying models.
// If there is an error, it logs it and then calls os.Exit(1).
func Finalize(i any) {
	ii, ok := i.(io.Closer)
	if !ok {
		return
	}
	if err := ii.Close(); err != nil {
		log.Fatal().Err(err).Send()
	}
}
