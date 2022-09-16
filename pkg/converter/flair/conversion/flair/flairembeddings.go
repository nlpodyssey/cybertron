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

type FlairEmbeddingsClass struct{}

type FlairEmbeddings struct {
	TokenEmbeddingsModule
	Name                      string
	FineTune                  bool
	StaticEmbeddings          bool
	IsForwardLm               bool
	CharsPerChunk             int
	embeddingLength           int
	PretrainedModelArchiveMap map[string]string
	LM                        *LanguageModel
}

var _ TokenEmbeddings = &FlairEmbeddings{}

func (FlairEmbeddingsClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("FlairEmbeddingsClass: unsupported arguments: %#v", args)
	}
	return &FlairEmbeddings{}, nil
}

func (f *FlairEmbeddings) EmbeddingLength() int {
	return f.embeddingLength
}

func (f *FlairEmbeddings) PyDictSet(k, v any) (err error) {
	if err := f.Module.PyDictSet(k, v); err == nil {
		return nil
	} else if !torch.IsUnexpectedModuleDictKey(err) {
		return fmt.Errorf("FlairEmbeddings: %w", err)
	}

	switch k {
	case "name":
		err = conversion.AssignAssertedType(v, &f.Name)
	case "fine_tune":
		err = conversion.AssignAssertedType(v, &f.FineTune)
	case "static_embeddings":
		err = conversion.AssignAssertedType(v, &f.StaticEmbeddings)
	case "is_forward_lm":
		err = conversion.AssignAssertedType(v, &f.IsForwardLm)
	case "chars_per_chunk":
		err = conversion.AssignAssertedType(v, &f.CharsPerChunk)
	case "_FlairEmbeddings__embedding_length":
		err = conversion.AssignAssertedType(v, &f.embeddingLength)
	case "PRETRAINED_MODEL_ARCHIVE_MAP":
		var d *types.Dict
		err = conversion.AssignAssertedType(v, &d)
		if err == nil {
			err = conversion.AssignDictToMap(d, &f.PretrainedModelArchiveMap)
		}
	case "detach", "cache": // TODO
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("FlairEmbeddings: dict key %q: %w", k, err)
	}
	return err
}

func (f *FlairEmbeddings) LoadStateDictEntry(k string, v any) (err error) {
	name, rest, _ := strings.Cut(k, ".")

	switch name {
	case "lm":
		f.LM, err = torch.GetSubModule[*LanguageModel](f.Module, name)
		if err == nil {
			err = f.LM.LoadStateDictEntry(rest, v)
		}
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("FlairEmbeddings: state dict key %q: %w", k, err)
	}
	return err
}
