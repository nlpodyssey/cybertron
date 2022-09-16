// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package torch

import (
	"fmt"
)

type LSTMClass struct{}

type LSTM struct {
	*RNNBase
}

func NewLSTM(c RNNBaseConfig) (*LSTM, error) {
	rb, err := NewRNNBase("LSTM", c)
	if err != nil {
		return nil, err
	}
	return &LSTM{RNNBase: rb}, nil
}

func (LSTMClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("LSTMClass: unsupported arguments: %#v", args)
	}
	return &LSTM{RNNBase: new(RNNBase)}, nil
}
