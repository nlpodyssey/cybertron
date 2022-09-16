// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gensim

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
)

type VocabClass struct{}

type Vocab struct {
	Count int
	Index int
}

func (VocabClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("VocabClass: want no arguments, got %d: %#v", len(args), args)
	}
	return &Vocab{}, nil
}

func (voc *Vocab) PyDictSet(k, v any) (err error) {
	switch k {
	case "count":
		err = conversion.AssignAssertedType(v, &voc.Count)
	case "index":
		err = conversion.AssignAssertedType(v, &voc.Index)
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("vocab: dict key %q: %w", k, err)
	}
	return err
}
