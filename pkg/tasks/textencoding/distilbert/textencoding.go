package distilbert

import (
	"context"
	"fmt"
	"path"
	"path/filepath"
	"strings"

	"github.com/nlpodyssey/spago/mat"

	"github.com/nlpodyssey/cybertron/pkg/models/distilbert"
	"github.com/nlpodyssey/cybertron/pkg/tasks/diskstoremode"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/cybertron/pkg/vocabulary"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/spago/nn"
)

var _ textencoding.Interface = &TextEncoding{}

// TextEncoding is a text encoding model.
type TextEncoding struct {
	// Model is the model used to answer questions.
	Model *distilbert.ModelForSequenceEncoding
	// Tokenizer is the tokenizer used to tokenize questions and passages.
	Tokenizer *wordpiecetokenizer.WordPieceTokenizer
	// doLowerCase is a flag indicating if the model should lowercase the input before tokenization.
	doLowerCase bool
	// embeddingsRepo is the repository used for loading embeddings.
	embeddingsRepo *diskstore.Repository
}

// LoadTextEncoding returns a TextEncoding loading the model, the embeddings and the tokenizer from a directory.
func LoadTextEncoding(modelPath string) (*TextEncoding, error) {
	vocab, err := vocabulary.NewFromFile(filepath.Join(modelPath, "vocab.txt"))
	if err != nil {
		return nil, fmt.Errorf("failed to load vocabulary for text encoding: %w", err)
	}
	tokenizer := wordpiecetokenizer.New(vocab)

	tokenizerConfig, err := distilbert.ConfigFromFile[distilbert.TokenizerConfig](path.Join(modelPath, "tokenizer_config.json"))
	if err != nil {
		return nil, fmt.Errorf("failed to load tokenizer config for text encoding: %w", err)
	}

	embeddingsRepo, err := diskstore.NewRepository(filepath.Join(modelPath, "repo"), diskstoremode.DefaultDiskStoreMode)
	if err != nil {
		return nil, fmt.Errorf("failed to load embeddings repository for text encoding: %w", err)
	}

	m, err := nn.LoadFromFile[*distilbert.ModelForSequenceEncoding](path.Join(modelPath, "spago_model.bin"))
	if err != nil {
		return nil, fmt.Errorf("failed to load distilbert model: %w", err)
	}

	err = m.DistilBert.SetEmbeddings(embeddingsRepo)
	if err != nil {
		return nil, fmt.Errorf("failed to set embeddings: %w", err)
	}

	return &TextEncoding{
		Model:          m,
		Tokenizer:      tokenizer,
		doLowerCase:    tokenizerConfig.DoLowerCase,
		embeddingsRepo: embeddingsRepo,
	}, nil
}

// Close finalizes the TextEncoding resources.
// It satisfies the interface io.Closer.
func (m *TextEncoding) Close() error {
	return m.embeddingsRepo.Close()
}

// Encode returns the dense encoded representation of the given text.
func (m *TextEncoding) Encode(_ context.Context, text string, poolingStrategy int) (textencoding.Response, error) {
	tokenized := m.tokenize(text)
	if l, max := len(tokenized), m.Model.DistilBert.Config.MaxPositionEmbeddings; l > max {
		return textencoding.Response{}, fmt.Errorf("%w: %d > %d", textencoding.ErrInputSequenceTooLong, l, max)
	}
	encoded, err := m.Model.Encode(tokenized, distilbert.PoolingStrategyType(poolingStrategy))
	if err != nil {
		return textencoding.Response{}, err
	}

	response := textencoding.Response{
		Vector: mat.CopyValue(encoded),
	}
	return response, nil
}

// tokenize returns the tokens of the given text (including padding tokens).
func (m *TextEncoding) tokenize(text string) []string {
	if m.doLowerCase {
		text = strings.ToLower(text)
	}
	cls := wordpiecetokenizer.DefaultClassToken
	sep := wordpiecetokenizer.DefaultSequenceSeparator
	return append([]string{cls}, append(tokenizers.GetStrings(m.Tokenizer.Tokenize(text)), sep)...)
}
