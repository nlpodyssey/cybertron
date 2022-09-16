// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
	"github.com/nlpodyssey/gopickle/pytorch"
)

type LinearClass struct{}

type Linear struct {
	Module
	InFeatures  int
	OutFeatures int
	Bias        *pytorch.Tensor
	Weight      *pytorch.Tensor
}

func (LinearClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("LinearClass: unsupported arguments: %#v", args)
	}
	return &Linear{}, nil
}

func (l *Linear) PyDictSet(k, v any) (err error) {
	if err := l.Module.PyDictSet(k, v); err == nil {
		return nil
	} else if !IsUnexpectedModuleDictKey(err) {
		return fmt.Errorf("linear: %w", err)
	}

	switch k {
	case "in_features":
		err = conversion.AssignAssertedType(v, &l.InFeatures)
	case "out_features":
		err = conversion.AssignAssertedType(v, &l.OutFeatures)
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("linear: dict key %q: %w", k, err)
	}
	return err
}

func (l *Linear) LoadStateDictEntry(k string, v any) (err error) {
	switch k {
	case "bias":
		l.Bias, err = AnyToTensor(v, []int{l.OutFeatures})
	case "weight":
		l.Weight, err = AnyToTensor(v, []int{l.OutFeatures, l.InFeatures})
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("linear: state dict key %q: %w", k, err)
	}
	return err
}
