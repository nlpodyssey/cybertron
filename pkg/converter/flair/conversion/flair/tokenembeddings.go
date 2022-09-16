package flair

import "github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion/torch"

type TokenEmbeddings interface {
	EmbeddingLength() int
	LoadStateDictEntry(k string, v any) error
	mustEmbedTokenEmbeddingsModule()
}

type TokenEmbeddingsModule struct {
	torch.Module
}

func (TokenEmbeddingsModule) mustEmbedTokenEmbeddingsModule() {}
