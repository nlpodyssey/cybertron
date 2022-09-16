// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/torch"
	"github.com/nlpodyssey/gopickle/pytorch"
)

type CRF struct {
	TagDictionary *Dictionary
	TagsetSize    int
	Transitions   *pytorch.Tensor
}

func (c *CRF) LoadStateDictEntry(k string, v any) (err error) {
	switch k {
	case "transitions":
		c.Transitions, err = torch.AnyToTensor(v, []int{c.TagsetSize, c.TagsetSize})
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("CRF: state dict key %q: %w", k, err)
	}
	return err
}
