// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package flair

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/gensim"
	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/torch"
	"github.com/nlpodyssey/gopickle/types"
	"github.com/nlpodyssey/spago/mat"
)

type WordEmbeddingsClass struct{}

type WordEmbeddings struct {
	TokenEmbeddingsModule
	Embeddings         string
	Name               string
	StaticEmbeddings   bool
	Embedding          *torch.Embedding
	Vocab              map[string]int
	InstanceParameters *types.Dict
	embeddingLength    int
}

var _ TokenEmbeddings = &WordEmbeddings{}

func (WordEmbeddingsClass) PyNew(args ...any) (any, error) {
	if len(args) != 0 {
		return nil, fmt.Errorf("WordEmbeddingsClass: unsupported arguments: %#v", args)
	}
	return &WordEmbeddings{}, nil
}

func (w *WordEmbeddings) EmbeddingLength() int {
	return w.embeddingLength
}

func (w *WordEmbeddings) PyDictSet(k, v any) (err error) {
	if err := w.Module.PyDictSet(k, v); err == nil {
		return nil
	} else if !torch.IsUnexpectedModuleDictKey(err) {
		return fmt.Errorf("WordEmbeddings: %w", err)
	}

	switch k {
	case "embeddings":
		err = conversion.AssignAssertedType(v, &w.Embeddings)
	case "get_cached_vec":
		// present on older models, can be ignored
	case "instance_parameters":
		err = conversion.AssignAssertedType(v, &w.InstanceParameters)
	case "name":
		err = conversion.AssignAssertedType(v, &w.Name)
	case "static_embeddings":
		err = conversion.AssignAssertedType(v, &w.StaticEmbeddings)
	case "precomputed_word_embeddings":
		var kv *gensim.KeyedVectors
		err = conversion.AssignAssertedType(v, &kv)
		if err == nil {
			err = w.setPrecomputedWordEmbeddings(kv)
		}
	case "_WordEmbeddings__embedding_length":
		err = conversion.AssignAssertedType(v, &w.embeddingLength)
	case "field":
		if v != nil {
			err = fmt.Errorf("only nil is supported, got %T: %#v", v, v)
		}
	default:
		err = fmt.Errorf("unexpected key with value %#v", v)
	}

	if err != nil {
		err = fmt.Errorf("WordEmbeddings: dict key %q: %w", k, err)
	}
	return err
}

func (w *WordEmbeddings) setPrecomputedWordEmbeddings(kv *gensim.KeyedVectors) error {
	vectors, err := kv.Vectors.SliceOfVectors()
	if err != nil {
		return fmt.Errorf("failed to convert Vectors: %w", err)
	}
	if len(vectors) > 0 && kv.VectorSize != vectors[0].Size() {
		return fmt.Errorf("VectorSize %d does not match actual vectors size %d", kv.VectorSize, vectors[0].Size())
	}
	vectors = append(vectors, mat.NewEmptyVecDense[float64](kv.VectorSize))
	w.Embedding = torch.EmbeddingFromPretrained(vectors, kv.VectorSize)

	w.Vocab = make(map[string]int, len(kv.Vocab))
	for k, v := range kv.Vocab {
		w.Vocab[k] = v.Index
	}

	return nil
}

func (w *WordEmbeddings) PyGetAttribute(name string) (value any, exists bool, err error) {
	switch name {
	case "get_cached_vec":
		// this ignores the get_cached_vec method when loading older versions
		// it is needed for compatibility reasons
		return nil, true, nil
	default:
		return nil, false, fmt.Errorf("WordEmbeddings: unexpected __getattribute__(%q)", name)
	}
}

func (w *WordEmbeddings) LoadStateDictEntry(string, any) error {
	return fmt.Errorf("WordEmbeddings: loading from state dict entry not implemented")
}
