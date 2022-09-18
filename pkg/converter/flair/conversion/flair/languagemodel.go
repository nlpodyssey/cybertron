// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"fmt"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/torch"
)

type LanguageModelClass struct{}

type LanguageModel struct {
	torch.Module
	Dictionary        *Dictionary
	IsForwardLm       bool
	Dropout           float64
	HiddenSize        int
	EmbeddingSize     int
	NLayers           int
	NOut              int
	DocumentDelimiter string

	Encoder *torch.SparseEmbedding
	Decoder *torch.Linear
	Proj    *torch.Linear
	RNN     *torch.LSTM
}

func (LanguageModelClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("LanguageModelClass: unsupported arguments: %#v", args)
	}
	return &LanguageModel{}, nil
}

func (l *LanguageModel) PyDictSet(k, v any) (err error) {
	if err := l.Module.PyDictSet(k, v); err == nil {
		return nil
	} else if !torch.IsUnexpectedModuleDictKey(err) {
		return fmt.Errorf("LanguageModel: %w", err)
	}

	switch k {
	case "dictionary":
		err = conversion.AssignAssertedType(v, &l.Dictionary)
	case "is_forward_lm":
		err = conversion.AssignAssertedType(v, &l.IsForwardLm)
	case "dropout":
		err = conversion.AssignAssertedType(v, &l.Dropout)
	case "hidden_size":
		err = conversion.AssignAssertedType(v, &l.HiddenSize)
	case "embedding_size":
		err = conversion.AssignAssertedType(v, &l.EmbeddingSize)
	case "nlayers":
		err = conversion.AssignAssertedType(v, &l.NLayers)
	case "nout":
		if v != nil {
			err = conversion.AssignAssertedType(v, &l.NOut)
		}
	case "proj":
		if v != nil {
			err = fmt.Errorf("only nil is supported, got %T: %#v", v, v)
		}
	case "hidden":
		if v != nil {
			err = fmt.Errorf("only nil is supported, got %T: %#v", v, v)
		}
	case "document_delimiter":
		err = conversion.AssignAssertedType(v, &l.DocumentDelimiter)
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("LanguageModel: dict key %q: %w", k, err)
	}
	return err
}

func (l *LanguageModel) LoadStateDictEntry(k string, v any) (err error) {
	name, rest, _ := strings.Cut(k, ".")

	switch name {
	case "proj":
		l.Proj, err = torch.GetSubModule[*torch.Linear](l.Module, name)
		if err == nil {
			err = l.Proj.LoadStateDictEntry(rest, v)
		}
	case "decoder":
		l.Decoder, err = torch.GetSubModule[*torch.Linear](l.Module, name)
		if err == nil {
			err = l.Decoder.LoadStateDictEntry(rest, v)
		}
	case "encoder":
		l.Encoder, err = torch.GetSubModule[*torch.SparseEmbedding](l.Module, name)
		if err == nil {
			err = l.Encoder.LoadStateDictEntry(rest, v)
		}
	case "rnn":
		l.RNN, err = torch.GetSubModule[*torch.LSTM](l.Module, name)
		if err == nil {
			err = l.RNN.LoadStateDictEntry(rest, v)
		}
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("LanguageModel: state dict key %q: %w", k, err)
	}
	return err
}
