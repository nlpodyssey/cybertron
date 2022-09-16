// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import (
	"errors"
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
	"github.com/nlpodyssey/gopickle/types"
)

type Module struct {
	Training              bool
	Parameters            *types.OrderedDict
	Buffers               *types.OrderedDict
	BackwardHooks         *types.OrderedDict
	ForwardHooks          *types.OrderedDict
	ForwardPreHooks       *types.OrderedDict
	StateDictHooks        *types.OrderedDict
	LoadStateDictPreHooks *types.OrderedDict
	Modules               *types.OrderedDict
}

func GetSubModule[T any](mod Module, name string) (v T, err error) {
	m, ok := mod.Modules.Get(name)
	if !ok {
		return v, fmt.Errorf("torch module not found: %q", name)
	}
	v, err = conversion.AssertType[T](m)
	if err != nil {
		return v, fmt.Errorf("torch module %q: %w", name, err)
	}
	return v, nil
}

func GetModuleParameter[T any](mod Module, name string) (v T, err error) {
	p, ok := mod.Parameters.Get(name)
	if !ok {
		return v, fmt.Errorf("torch parameter not found: %q", name)
	}
	v, err = conversion.AssertType[T](p)
	if err != nil {
		return v, fmt.Errorf("torch parameter %q: %w", name, err)
	}
	return v, nil
}

var ErrUnexpectedModuleDictKey = errors.New("unexpected dict key")

func IsUnexpectedModuleDictKey(err error) bool {
	return err != nil && errors.Is(err, ErrUnexpectedModuleDictKey)
}

func (m *Module) PyDictSet(k, v any) (err error) {
	switch k {
	case "training":
		err = conversion.AssignAssertedType(v, &m.Training)
	case "_parameters":
		err = conversion.AssignAssertedType(v, &m.Parameters)
	case "_buffers":
		err = conversion.AssignAssertedType(v, &m.Buffers)
	case "_backward_hooks":
		err = conversion.AssignAssertedType(v, &m.BackwardHooks)
	case "_forward_hooks":
		err = conversion.AssignAssertedType(v, &m.ForwardHooks)
	case "_forward_pre_hooks":
		err = conversion.AssignAssertedType(v, &m.ForwardPreHooks)
	case "_state_dict_hooks":
		err = conversion.AssignAssertedType(v, &m.StateDictHooks)
	case "_load_state_dict_pre_hooks":
		err = conversion.AssignAssertedType(v, &m.LoadStateDictPreHooks)
	case "_modules":
		err = conversion.AssignAssertedType(v, &m.Modules)
	case "_backend":
		if v != nil {
			err = fmt.Errorf("only nil is supported, got %T: %#v", v, v)
		}
	default:
		err = fmt.Errorf("%w with value %#v", ErrUnexpectedModuleDictKey, v)
	}

	if err != nil {
		err = fmt.Errorf("torch module: dict key %q: %w", k, err)
	}
	return err
}
