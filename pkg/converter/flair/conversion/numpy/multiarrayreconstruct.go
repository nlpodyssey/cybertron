// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package numpy

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
	"github.com/nlpodyssey/gopickle/types"
)

type MultiarrayReconstruct struct{}

func (MultiarrayReconstruct) Call(args ...any) (any, error) {
	if len(args) != 3 {
		return nil, fmt.Errorf("MultiarrayReconstruct: want 3 args, got %d: %#v", len(args), args)
	}

	subType, err := conversion.AssertType[types.PyNewable](args[0])
	if err != nil {
		return nil, fmt.Errorf("MultiarrayReconstruct: 1st arg (subtype): %w", err)
	}

	shape, err := conversion.AssertType[*types.Tuple](args[1])
	if err != nil {
		return nil, fmt.Errorf("MultiarrayReconstruct: 2nd arg (shape): %w", err)
	}

	dataType, err := conversion.AssertType[[]byte](args[2])
	if err != nil {
		return nil, fmt.Errorf("MultiarrayReconstruct: 3rd arg (dtype): %w", err)
	}

	return subType.PyNew(shape, string(dataType))
}
