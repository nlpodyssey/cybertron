// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import (
	"fmt"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
)

type RNN interface {
	HiddenSize() int
	LoadStateDictEntry(k string, v any) error
	mustEmbedRNNBase()
}

type RNNBaseConfig struct {
	InputSize     int
	HiddenSize    int
	NumLayers     int
	Bias          bool
	BatchFirst    bool
	Dropout       float64
	Bidirectional bool
	ProjSize      int
}

type RNNBase struct {
	Module
	Mode             string
	InputSize        int
	hiddenSize       int
	NumLayers        int
	Bias             bool
	BatchFirst       bool
	Dropout          float64
	Bidirectional    bool
	ProjSize         int
	FlatWeightsNames []string
	AllWeights       [][]string
	Parameters       map[string]*Parameter
}

var _ RNN = &RNNBase{}

func NewRNNBase(mode string, c RNNBaseConfig) (*RNNBase, error) {
	switch {
	case c.Dropout < 0 || c.Dropout > 1:
		return nil, fmt.Errorf("RNNBase: want dropout in range [0, 1], got %f", c.Dropout)
	case c.ProjSize < 0:
		return nil, fmt.Errorf("RNNBase: want ProjSize >= 0, got %d", c.ProjSize)
	case c.ProjSize >= c.HiddenSize:
		return nil, fmt.Errorf("RNNBase: want ProjSize >= HiddenSize (%d), got %d", c.HiddenSize, c.ProjSize)
	}

	var gateSize int
	switch mode {
	case "LSTM":
		gateSize = 4 * c.HiddenSize
	default:
		return nil, fmt.Errorf("RNNBase: invalid or unimplemented mode %q", mode)
	}

	r := &RNNBase{
		Mode:             mode,
		InputSize:        c.InputSize,
		hiddenSize:       c.HiddenSize,
		NumLayers:        c.NumLayers,
		Bias:             c.Bias,
		BatchFirst:       c.BatchFirst,
		Dropout:          c.Dropout,
		Bidirectional:    c.Bidirectional,
		ProjSize:         c.ProjSize,
		FlatWeightsNames: make([]string, 0),
		AllWeights:       make([][]string, 0),
		Parameters:       make(map[string]*Parameter),
	}

	numDirections := 1
	if c.Bidirectional {
		numDirections = 2
	}

	realHiddenSize := c.HiddenSize
	if c.ProjSize > 0 {
		realHiddenSize = c.ProjSize
	}

	for layer := 0; layer < c.NumLayers; layer++ {
		for direction := 0; direction < numDirections; direction++ {
			layerInputSize := realHiddenSize * numDirections
			if layer == 0 {
				layerInputSize = c.InputSize
			}
			wIH := &Parameter{
				Data:         &pytorch.Tensor{Size: []int{gateSize, layerInputSize}},
				RequiresGrad: true,
			}
			wHH := &Parameter{
				Data:         &pytorch.Tensor{Size: []int{gateSize, realHiddenSize}},
				RequiresGrad: true,
			}
			bIH := &Parameter{
				Data:         &pytorch.Tensor{Size: []int{gateSize}},
				RequiresGrad: true,
			}

			// Second bias vector included for CuDNN compatibility. Only one
			// bias vector is needed in standard definition.
			bHH := &Parameter{
				Data:         &pytorch.Tensor{Size: []int{gateSize}},
				RequiresGrad: true,
			}
			var layerParams []*Parameter
			if r.ProjSize == 0 {
				if c.Bias {
					layerParams = []*Parameter{wIH, wHH, bIH, bHH}
				} else {
					layerParams = []*Parameter{wIH, wHH}
				}
			} else {
				wHR := &Parameter{
					Data:         &pytorch.Tensor{Size: []int{c.ProjSize, c.HiddenSize}},
					RequiresGrad: true,
				}
				if c.Bias {
					layerParams = []*Parameter{wIH, wHH, bIH, bHH, wHR}
				} else {
					layerParams = []*Parameter{wIH, wHH, wHR}
				}
			}

			suffix := ""
			if direction == 1 {
				suffix = "_reverse"
			}
			paramNames := []string{"weight_ih_l%d%s", "weight_hh_l%d%s"}
			if c.Bias {
				paramNames = append(paramNames, "bias_ih_l%d%s")
				paramNames = append(paramNames, "bias_hh_l%d%s")
			}
			if r.ProjSize > 0 {
				paramNames = append(paramNames, "weight_hr_l%d%s")
			}
			for i, format := range paramNames {
				paramNames[i] = fmt.Sprintf(format, layer, suffix)
			}

			for i, name := range paramNames {
				r.Parameters[name] = layerParams[i]
				// TODO: module.parameters?
			}

			r.FlatWeightsNames = append(r.FlatWeightsNames, paramNames...)
			r.AllWeights = append(r.AllWeights, paramNames)
		}
	}

	// TODO: self._flat_weights = [(lambda wn: getattr(self, wn) if hasattr(self, wn) else None)(wn) for wn in self._flat_weights_names]
	// TODO: self.flatten_parameters()
	//
	// TODO: self.reset_parameters()

	return r, nil
}

func (*RNNBase) mustEmbedRNNBase() {}

func (r *RNNBase) HiddenSize() int {
	return r.hiddenSize
}

func (r *RNNBase) PyDictSet(k, v any) (err error) {
	if err := r.Module.PyDictSet(k, v); err == nil {
		return nil
	} else if !IsUnexpectedModuleDictKey(err) {
		return fmt.Errorf("RNNBase: %w", err)
	}

	switch k {
	case "mode":
		err = conversion.AssignAssertedType(v, &r.Mode)
	case "input_size":
		err = conversion.AssignAssertedType(v, &r.InputSize)
	case "hidden_size":
		err = conversion.AssignAssertedType(v, &r.hiddenSize)
	case "num_layers":
		err = conversion.AssignAssertedType(v, &r.NumLayers)
	case "bias":
		err = conversion.AssignAssertedType(v, &r.Bias)
	case "batch_first":
		err = conversion.AssignAssertedType(v, &r.BatchFirst)
	case "dropout":
	//	err = conversion.AssignAssertedType(v, &r.Dropout)
	case "dropout_state":
	case "bidirectional":
		err = conversion.AssignAssertedType(v, &r.Bidirectional)
	case "_all_weights":
		err = r.convertAndSetAllWeights(v)
	case "_data_ptrs", "_param_buf_size":
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("RNNBase mode %q: dict key %q: %w", r.Mode, k, err)
	}
	return err
}

func (r *RNNBase) convertAndSetAllWeights(v any) error {
	outer, err := conversion.AssertType[*types.List](v)
	if err != nil {
		return err
	}

	r.AllWeights = make([][]string, outer.Len())

	for i, outerItem := range *outer {
		inner, err := conversion.AssertType[*types.List](outerItem)
		if err == nil {
			err = conversion.AssignListToSlice(inner, &r.AllWeights[i])
		}
		if err != nil {
			return fmt.Errorf("inner item at index %d: %w", i, err)
		}
	}

	return nil
}

func (r *RNNBase) LoadStateDictEntry(k string, v any) (err error) {
	switch {
	case strings.HasPrefix(k, "weight_") || strings.HasPrefix(k, "bias_"):
		err = r.loadStateDictParameter(k, v)
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("RNNBase mode %q: state dict key %q: %w", r.Mode, k, err)
	}
	return err
}

func (r *RNNBase) loadStateDictParameter(k string, v any) (err error) {
	p, ok := r.Parameters[k]
	if !ok {
		p, err = GetModuleParameter[*Parameter](r.Module, k)
		if err != nil {
			t, err := GetModuleParameter[*pytorch.Tensor](r.Module, k)
			if err != nil {
				return err
			}
			p = &Parameter{
				Data:         t,
				RequiresGrad: true,
			}
		}
	}

	if r.Parameters == nil {
		r.Parameters = make(map[string]*Parameter)
	}
	t, err := AnyToTensor(v, p.Data.Size)
	if err != nil {
		return err
	}

	p.Data = t
	r.Parameters[k] = p
	return nil
}
