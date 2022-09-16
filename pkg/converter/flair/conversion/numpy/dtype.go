// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package numpy

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
	"github.com/nlpodyssey/gopickle/types"
)

type DTypeClass struct{}

type DType struct {
	DType     string
	Align     bool
	Copy      bool
	ByteOrder string
	State     *types.Tuple
}

func (DTypeClass) Call(args ...any) (any, error) {
	if len(args) == 0 || len(args) > 3 {
		return nil, fmt.Errorf("DTypeClass: want 1 to 3 arguments, got %d: %#v", len(args), args)
	}

	dt := new(DType)

	err := conversion.AssignAssertedType(args[0], &dt.DType)
	if err != nil {
		return nil, fmt.Errorf("DTypeClass: 1st arg (dtype): %w", err)
	}

	if len(args) == 1 {
		return dt, nil
	}

	switch v := args[1].(type) {
	case int:
		dt.Align = v != 0
	case bool:
		dt.Align = v
	default:
		return nil, fmt.Errorf("DTypeClass: 2nd arg (align): %w", err)
	}

	if len(args) == 2 {
		return dt, nil
	}

	switch v := args[2].(type) {
	case int:
		dt.Copy = v != 0
	case bool:
		dt.Copy = v
	default:
		return nil, fmt.Errorf("DTypeClass: 3rd arg (copy): %w", err)
	}

	return dt, nil
}

func (d *DType) PySetState(state any) error {
	err := conversion.AssignAssertedType(state, &d.State)
	if err != nil {
		return fmt.Errorf("DType: failed to set state: %w", err)
	}

	if d.State.Len() >= 2 {
		err = conversion.AssignAssertedType(d.State.Get(1), &d.ByteOrder)
		if err != nil {
			return fmt.Errorf("DType: 2nd state Tuple item (byteorder): %w", err)
		}
	}

	return nil
}
