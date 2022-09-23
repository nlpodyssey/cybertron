package colbert

import (
	"fmt"
	"path"
	"path/filepath"
	"strings"

	"github.com/nlpodyssey/cybertron/pkg/models/bert"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers"
	"github.com/nlpodyssey/cybertron/pkg/tokenizers/wordpiecetokenizer"
	"github.com/nlpodyssey/cybertron/pkg/vocabulary"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/spago/nn"
)

const SpecialDocumentMarker = "[unused1]"

const SpecialQueryMarker = "[unused0]"

type DocumentScorer struct {
	Model     *bert.ColbertModel
	Tokenizer *wordpiecetokenizer.WordPieceTokenizer
}

func LoadDocumentScorer(modelPath string) (*DocumentScorer, error) {
	vocab, err := vocabulary.NewFromFile(filepath.Join(modelPath, "vocab.txt"))
	if err != nil {
		return nil, fmt.Errorf("failed to load vocabulary: %w", err)
	}

	tokenizer := wordpiecetokenizer.New(vocab)

	embeddingsRepo, err := diskstore.NewRepository(filepath.Join(modelPath, "repo"), diskstore.ReadOnlyMode)
	if err != nil {
		return nil, fmt.Errorf("failed to load embeddings repository: %w", err)
	}

	m, err := nn.LoadFromFile[*bert.ColbertModel](path.Join(modelPath, "spago_model.bin"))
	if err != nil {
		return nil, fmt.Errorf("failed to load colbert model: %w", err)
	}

	err = m.Bert.SetEmbeddings(embeddingsRepo)
	if err != nil {
		return nil, fmt.Errorf("failed to set embeddings: %w", err)
	}
	return &DocumentScorer{
		Model:     m,
		Tokenizer: tokenizer,
	}, nil
}

func (r *DocumentScorer) encode(text string, specialMarker string) []ag.Node {
	tokens := r.Tokenizer.Tokenize(strings.ToLower(text))

	stringTokens := tokenizers.GetStrings(tokens)
	stringTokens = append([]string{wordpiecetokenizer.DefaultClassToken, specialMarker}, stringTokens...)
	stringTokens = append(stringTokens, wordpiecetokenizer.DefaultSequenceSeparator)
	embeddings := normalizeEmbeddings(r.Model.Forward(stringTokens))
	return filterEmbeddings(embeddings, stringTokens)
}

func (r *DocumentScorer) EncodeDocument(text string) []ag.Node {
	return r.encode(text, SpecialDocumentMarker)
}

func (r *DocumentScorer) EncodeQuery(text string) []ag.Node {
	return r.encode(text, SpecialQueryMarker)
}

func (r *DocumentScorer) ScoreDocument(query []ag.Node, document []ag.Node) ag.Node {
	var score ag.Node
	score = ag.Scalar(0.0)
	for i, q := range query {
		if i < 3 || i > len(query)-1 {
			continue // don't take special tokens into consideration
		}
		score = ag.Add(score, r.maxSimilarity(q, document))
	}
	return score
}

func (r *DocumentScorer) maxSimilarity(query ag.Node, document []ag.Node) ag.Node {
	var max ag.Node
	max = ag.Scalar(0.0)
	for i, d := range document {
		if i < 3 || i > len(document)-1 {
			continue // don't take special tokens into consideration
		}
		sim := ag.Dot(query, d)
		max = ag.Max(max, sim)
	}
	return max
}

func normalizeEmbeddings(embeddings []ag.Node) []ag.Node {
	// Perform l2 normalization of each embedding
	normalized := make([]ag.Node, len(embeddings))
	for i, e := range embeddings {
		normalized[i] = ag.DivScalar(e, ag.Sqrt(ag.ReduceSum(ag.Square(e))))
	}
	return normalized
}

func isPunctuation(token string) bool {
	return token == "." || token == "," || token == "!" || token == "?" ||
		token == ":" || token == ";" || token == "-" || token == "'" ||
		token == "\"" || token == "(" || token == ")" || token == "[" ||
		token == "]" || token == "{" || token == "}" || token == "*" ||
		token == "&" || token == "%" || token == "$" || token == "#" ||
		token == "@" || token == "=" || token == "+" ||
		token == "_" || token == "~" || token == "/" || token == "\\" ||
		token == "|" || token == "`" || token == "^" || token == ">" ||
		token == "<"
}

func filterEmbeddings(embeddings []ag.Node, tokens []string) []ag.Node {
	filtered := make([]ag.Node, 0, len(embeddings))
	for i, e := range embeddings {
		if isPunctuation(tokens[i]) {
			continue
		}
		filtered = append(filtered, e)
	}
	return filtered
}
