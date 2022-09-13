// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tokenclassification

import "context"

type AggregationStrategy string

const (
	// AggregationStrategyNone - Every token gets classified without further aggregation.
	AggregationStrategyNone AggregationStrategy = "none"

	// AggregationStrategySimple - Entities are grouped according to the IOB annotation schema.
	AggregationStrategySimple AggregationStrategy = "simple"
)

type Parameters struct {
	AggregationStrategy AggregationStrategy
}

// Interface defines the main functions for token classification task.
type Interface interface {
	// Classify returns the classification of the given example.
	Classify(ctx context.Context, text string, parameters Parameters) (Response, error)
}

// Token is a labeled text token.
type Token struct {
	Text  string
	Start int
	End   int
	Label string
	Score float64
}

// Response contains the response from token classification.
type Response struct {
	Tokens []Token
}
