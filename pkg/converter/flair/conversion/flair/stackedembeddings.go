// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"fmt"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/torch"
	"github.com/nlpodyssey/gopickle/types"
)

type StackedEmbeddingsClass struct{}

type StackedEmbeddings struct {
	TokenEmbeddingsModule
	Embeddings       []TokenEmbeddings
	Name             string
	StaticEmbeddings bool
	EmbeddingType    string
	embeddingLength  int
}

var _ TokenEmbeddings = &StackedEmbeddings{}

func (StackedEmbeddingsClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("StackedEmbeddingsClass: unsupported arguments: %#v", args)
	}
	return &StackedEmbeddings{}, nil
}

func (s *StackedEmbeddings) EmbeddingLength() int {
	return s.embeddingLength
}

func (s *StackedEmbeddings) PyDictSet(k, v any) (err error) {
	if err := s.Module.PyDictSet(k, v); err == nil {
		return nil
	} else if !torch.IsUnexpectedModuleDictKey(err) {
		return fmt.Errorf("StackedEmbeddings: %w", err)
	}

	switch k {
	case "embeddings":
		var l *types.List
		err = conversion.AssignAssertedType(v, &l)
		if err == nil {
			err = conversion.AssignListToSlice(l, &s.Embeddings)
		}
	case "name":
		err = conversion.AssignAssertedType(v, &s.Name)
	case "static_embeddings":
		err = conversion.AssignAssertedType(v, &s.StaticEmbeddings)
	case "_StackedEmbeddings__embedding_type":
		err = conversion.AssignAssertedType(v, &s.EmbeddingType)
	case "_StackedEmbeddings__embedding_length":
		err = conversion.AssignAssertedType(v, &s.embeddingLength)
	case "detach", "cache": // TODO
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("StackedEmbeddings: dict key %q: %w", k, err)
	}
	return err
}

func (s *StackedEmbeddings) LoadStateDictEntry(k string, v any) (err error) {
	name, rest, _ := strings.Cut(k, ".")

	switch {
	case strings.HasPrefix(name, "list_embedding_"):
		var te TokenEmbeddings
		te, err = torch.GetSubModule[TokenEmbeddings](s.Module, name)
		if err == nil {
			err = te.LoadStateDictEntry(rest, v)
		}
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("StackedEmbeddings: state dict key %q: %w", k, err)
	}
	return err
}
