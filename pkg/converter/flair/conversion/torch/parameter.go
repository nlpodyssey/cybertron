// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
)

type Parameter struct {
	Data         *pytorch.Tensor
	RequiresGrad bool
}

type RebuildParameter struct{}

func (RebuildParameter) Call(args ...any) (any, error) {
	if len(args) != 3 {
		return nil, fmt.Errorf("RebuildParameter: want 3 args, got %#v", args)
	}

	p := new(Parameter)

	err := conversion.AssignAssertedType(args[0], &p.Data)
	if err != nil {
		return nil, fmt.Errorf("RebuildParameter: 1st arg (data): %w", err)
	}

	err = conversion.AssignAssertedType(args[1], &p.RequiresGrad)
	if err != nil {
		return nil, fmt.Errorf("RebuildParameter: 2nd arg (requires_grad): %w", err)
	}

	// The third parameter is for backwards compatibility: the general
	// expectation is that backward_hooks is an empty OrderedDict.
	bh, err := conversion.AssertType[*types.OrderedDict](args[2])
	if err != nil {
		return nil, fmt.Errorf("RebuildParameter: 3rd arg (backward_hooks): %w", err)
	}
	if l := bh.Len(); l != 0 {
		return nil, fmt.Errorf("RebuildParameter: 3rd arg (backward_hooks): want empty dict, got length %d", l)
	}

	return p, nil
}
