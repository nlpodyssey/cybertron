// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import (
	"fmt"
	"reflect"

	"github.com/nlpodyssey/gopickle/pytorch"
)

func AnyToTensor(v any, mustHaveSize []int) (*pytorch.Tensor, error) {
	t, ok := v.(*pytorch.Tensor)
	if !ok {
		return nil, fmt.Errorf("expected *Tensor value, actual %T: %#v", v, v)
	}
	if mustHaveSize != nil && !reflect.DeepEqual(t.Size, mustHaveSize) {
		return nil, fmt.Errorf("expected Tensor with size %v, actual size %v", mustHaveSize, t.Size)
	}
	return t, nil
}
