package colbert

import (
	"os"
	"sort"
	"testing"

	"github.com/nlpodyssey/spago/ag"
	"github.com/stretchr/testify/require"
)

func TestDocumentScorer_ScoreDocument(t *testing.T) {

	tests := []struct {
		name        string
		query       string
		documents   []string
		wantRanking []int
		wantScores  []float64
	}{
		{
			name:        "test1",
			query:       "hello world",
			documents:   []string{"hello world"},
			wantRanking: []int{0},
			wantScores:  []float64{1.0},
		},
		{
			name:  "test2",
			query: "In which year was the first iPhone released?",
			documents: []string{"The first Nokia phone was released in 1987.",
				"The iPhone 3G was released in 2008.",
				"The original iPhone was first sold in 2007."},
			wantRanking: []int{2, 0, 1},
		},
	}
	// Set the directory where the colbert model is stored here:
	ColbertModelDir := "testdata/colbert"
	// if dir does not exist, skip test
	if _, err := os.Stat(ColbertModelDir); os.IsNotExist(err) {
		t.Skip("Colbert model directory does not exist, skipping test")
	}

	scorer, err := LoadDocumentScorer(ColbertModelDir)
	require.NoError(t, err)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			query := scorer.EncodeQuery(tt.query)
			var scores []float64
			for i, doc := range tt.documents {
				document := scorer.EncodeDocument(doc)
				score := scorer.ScoreDocument(query, document)
				// Normalize the score by the length of the non-special tokens of  query
				// (this is not in the original paper btw, but it makes sense to me)
				score = ag.Div(score, ag.Scalar(float64(len(query)-3)))
				if tt.wantScores != nil {
					require.InDelta(t, tt.wantScores[i], score.Value().Data().F64()[0], 0.01)
				}
				scores = append(scores, score.Value().Data().F64()[0])
			}
			ranking := rank(scores)
			require.Equal(t, tt.wantRanking, ranking)
		})
	}
}

func rank(scores []float64) []int {
	var ranking []int
	for i := range scores {
		ranking = append(ranking, i)
	}
	sort.SliceStable(ranking, func(i, j int) bool {
		return scores[ranking[i]] > scores[ranking[j]]
	})
	return ranking
}
