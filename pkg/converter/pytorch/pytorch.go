// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pytorch

import (
	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
	"github.com/nlpodyssey/spago/mat/float"
)

// ParamsProvider is a provider of parameters for a PyTorch model.
type ParamsProvider[T float.DType] struct {
	paramsData    map[string][]T
	nameMapping   MappingFunc
	preProcessing PreProcessingFunc[T]
}

// MappingFunc is a function that maps a parameter name to another name.
type MappingFunc func(name string) string

// PreProcessingFunc is a function that pre-processes parameters before the conversion.
type PreProcessingFunc[T float.DType] func(params *ParamsProvider[T]) error

// NewParamsProvider returns a new ParamsProvider.
func NewParamsProvider[T float.DType]() *ParamsProvider[T] {
	return &ParamsProvider[T]{
		paramsData: make(map[string][]T),
	}
}

// WithNameMapping sets the name mapping function.
func (p *ParamsProvider[T]) WithNameMapping(fn MappingFunc) *ParamsProvider[T] {
	p.nameMapping = fn
	return p
}

// WithPreProcessing sets the parameters pre-processing function.
func (p *ParamsProvider[T]) WithPreProcessing(fn PreProcessingFunc[T]) *ParamsProvider[T] {
	p.preProcessing = fn
	return p
}

// Load loads parameters from a PyTorch model.
func (p *ParamsProvider[T]) Load(filename string) error {
	result, err := pytorch.Load(filename)
	if err != nil {
		return err
	}
	fn := func(name string, tensor *pytorch.Tensor) {
		if _, ok := tensor.Source.(*pytorch.FloatStorage); ok {
			if p.nameMapping != nil {
				name = p.nameMapping(name)
			}
			p.paramsData[name] = data[T](tensor)
		}
	}
	switch r := result.(type) {
	case *types.OrderedDict:
		p.yieldOrderedDict(r, fn)
	case *types.Dict:
		p.yieldDict(r, fn)
	}
	err = p.preProcessing(p)
	return err
}

func (p *ParamsProvider[T]) yieldOrderedDict(dict *types.OrderedDict, fn func(name string, tensor *pytorch.Tensor)) {
	for key, entry := range dict.Map {
		fn(key.(string), entry.Value.(*pytorch.Tensor))
	}
}

func (p *ParamsProvider[T]) yieldDict(dict *types.Dict, fn func(name string, tensor *pytorch.Tensor)) {
	for _, entry := range *dict {
		fn(entry.Key.(string), entry.Value.(*pytorch.Tensor))
	}
}

// Pop returns a parameter with the given name and remove it from the params list.
func (p *ParamsProvider[T]) Pop(name string) []T {
	data := p.paramsData[name]
	delete(p.paramsData, name)
	return data
}

// Get returns a parameter with the given name.
func (p *ParamsProvider[T]) Get(name string) []T {
	return p.paramsData[name]
}

// Delete deletes a parameter with the given name.
func (p *ParamsProvider[T]) Delete(name string) {
	delete(p.paramsData, name)
}

// Set sets a parameter with the given name.
func (p *ParamsProvider[T]) Set(name string, data []T) {
	p.paramsData[name] = data
}

// data returns the underlying values of a PyTorch tensor as a T slice.
// It returns the data using the row-major representation, possibly converting column-major order to row-major order.
func data[T float.DType](t *pytorch.Tensor) []T {
	if len(t.Size) == 0 || len(t.Size) > 2 {
		panic("gopickleutils: number of sizes not supported")
	}
	size := t.Size[0]
	if len(t.Size) > 1 {
		size *= t.Size[1]
	}
	orig := t.Source.(*pytorch.FloatStorage).Data[t.StorageOffset : t.StorageOffset+size]
	data := make([]T, len(orig))

	if len(t.Size) == 1 || t.Size[1] == 1 || t.Size[0] == 1 || t.Stride[1] == 1 {
		for i, val := range orig {
			data[i] = T(val)
		}
		return data
	}

	s0, s1 := t.Size[1], t.Size[0]
	for i := 0; i < s0; i++ {
		for j := 0; j < s1; j++ {
			data[i+j*s0] = T(orig[j+i*s1])
		}
	}
	return data
}

// Iterate iterates over all the parameters in the provider.
func (p *ParamsProvider[T]) Iterate(fn func(name string, data []T) error) error {
	for name, data := range p.paramsData {
		if err := fn(name, data); err != nil {
			return err
		}
	}
	return nil
}
