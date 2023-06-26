package distilbert

import (
	"github.com/nlpodyssey/spago/ag"
	emb "github.com/nlpodyssey/spago/embeddings"
	"github.com/nlpodyssey/spago/embeddings/store"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/linear"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

// Embeddings implements a DistilBert input embedding module.
type Embeddings struct {
	nn.Module
	Tokens    *emb.Model[string]
	Positions *emb.Model[int]
	Norm      *layernorm.Model
	Projector *linear.Model
	Config    Config
}

// NewEmbeddings returns a new DistilBert input embedding module.
func NewEmbeddings[T float.DType](c Config, repo store.Repository) *Embeddings {
	var projector *linear.Model = nil
	if c.EmbeddingsSize != c.HiddenSize {
		projector = linear.New[T](c.EmbeddingsSize, c.HiddenSize)
	}

	return &Embeddings{
		Tokens: emb.New[T, string](emb.Config{
			Size:      c.EmbeddingsSize,
			StoreName: c.Cybertron.TokensStoreName,
			Trainable: c.Cybertron.Training,
		}, repo),
		Positions: emb.New[T, int](emb.Config{
			Size:      c.EmbeddingsSize,
			StoreName: c.Cybertron.PositionsStoreName,
			Trainable: c.Cybertron.Training,
		}, repo),
		Norm:      layernorm.New[T](c.EmbeddingsSize, 1e-5),
		Projector: projector,
		Config:    c,
	}
}

// Encode performs the DistilBert input encoding.
func (m *Embeddings) Encode(tokens []string) []ag.Node {
	var (
		encoded   = m.Tokens.Encode(tokens)
		positions = m.Positions.Encode(indices(len(tokens)))
	)

	for i := 0; i < len(tokens); i++ {
		encoded[i] = ag.Sum(encoded[i], positions[i])
	}
	return m.useDropout(m.Norm.Forward(encoded...))
}

// useDropout returns the output of the dropout if it is not nil, otherwise the input.
func (m *Embeddings) useDropout(xs []ag.Node) []ag.Node {
	if m.Projector == nil {
		return xs
	}
	return m.Projector.Forward(xs...)
}

// indices returns a slice of the given size, where each element has
// the same value of its own index position.
func indices(size int) []int {
	idx := make([]int, size)
	for i := range idx {
		idx[i] = i
	}
	return idx
}
