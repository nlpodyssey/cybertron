// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/collections"
	"github.com/nlpodyssey/gopickle/types"
)

type DictionaryClass struct{}

type Dictionary struct {
	Item2Idx           map[string]int
	Idx2Item           []string
	Item2IdxNotEncoded *collections.DefaultDict
	MultiLabel         bool
}

func (DictionaryClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("DictionaryClass: unsupported arguments: %#v", args)
	}
	return &Dictionary{}, nil
}

func (d *Dictionary) PyDictSet(k, v any) (err error) {
	switch k {
	case "item2idx":
		var dict *types.Dict
		err = conversion.AssignAssertedType(v, &dict)
		if err == nil {
			err = d.setItem2Idx(dict)
		}
	case "idx2item":
		var l *types.List
		err = conversion.AssignAssertedType(v, &l)
		if err == nil {
			err = d.setIdx2Item(l)
		}
	case "item2idx_not_encoded":
		err = conversion.AssignAssertedType(v, &d.Item2IdxNotEncoded)
	case "multi_label":
		err = conversion.AssignAssertedType(v, &d.MultiLabel)
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("dictionary: dict key %q: %w", k, err)
	}
	return err
}

func (d *Dictionary) Size() int {
	return len(d.Idx2Item)
}

func (d *Dictionary) setItem2Idx(pyDict *types.Dict) error {
	d.Item2Idx = make(map[string]int, pyDict.Len())
	for _, kv := range *pyDict {
		k, ok := kv.Key.([]byte)
		if !ok {
			return fmt.Errorf("dictionary: item2idx: want key type []byte, got %#v", kv.Key)
		}
		v, ok := kv.Value.(int)
		if !ok {
			return fmt.Errorf("dictionary: item2idx: want value type int, got %#v", kv.Value)
		}
		d.Item2Idx[string(k)] = v
	}
	return nil
}

func (d *Dictionary) setIdx2Item(pyList *types.List) error {
	d.Idx2Item = make([]string, pyList.Len())
	for i, pv := range *pyList {
		v, ok := pv.([]byte)
		if !ok {
			return fmt.Errorf("dictionary: idx2item: want item type []byte, got %#v", pv)
		}
		d.Idx2Item[i] = string(v)
	}
	return nil
}
