// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
)

type DropoutClass struct{}

type Dropout struct {
	Module
	P       float64
	InPlace bool
}

func (DropoutClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("DropoutClass: unsupported arguments: %#v", args)
	}
	return &Dropout{}, nil
}

func (d *Dropout) PyDictSet(k, v any) (err error) {
	if err := d.Module.PyDictSet(k, v); err == nil {
		return nil
	} else if !IsUnexpectedModuleDictKey(err) {
		return fmt.Errorf("dropout: %w", err)
	}

	switch k {
	case "p":
		err = conversion.AssignAssertedType(v, &d.P)
	case "inplace":
		err = conversion.AssignAssertedType(v, &d.InPlace)
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("dropout: dict key %q: %w", k, err)
	}
	return err
}
