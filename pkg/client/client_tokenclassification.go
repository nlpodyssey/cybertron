// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package client

import (
	"context"
	"fmt"
	"time"

	tokenclassificationv1 "github.com/nlpodyssey/cybertron/pkg/server/gen/proto/go/tokenclassification/v1"
	"github.com/nlpodyssey/cybertron/pkg/tasks/tokenclassification"
)

var _ tokenclassification.Interface = &clientForTokenClassification{}

// clientForTextClassification is a client for token classification implementing tokenclassification.Interface
type clientForTokenClassification struct {
	// target is the server endpoint.
	target string
	// opts is the gRPC options for the client.
	opts Options
}

// NewClientForTokenClassification creates a new client for token classification.
func NewClientForTokenClassification(target string, opts Options) tokenclassification.Interface {
	return &clientForTokenClassification{
		target: target,
		opts:   opts,
	}
}

// Classify classifies the given text.
func (c *clientForTokenClassification) Classify(ctx context.Context, text string, parameters tokenclassification.Parameters) (tokenclassification.Response, error) {
	conn, err := Dial(ctx, c.target, c.opts)
	if err != nil {
		return tokenclassification.Response{}, fmt.Errorf("failed to dial %q: %w", c.target, err)
	}
	cc := tokenclassificationv1.NewTokenClassificationServiceClient(conn)

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	response, err := cc.Classify(ctx, &tokenclassificationv1.ClassifyRequest{
		Input:               text,
		AggregationStrategy: grpcAggregationStrategy(parameters.AggregationStrategy),
	})
	if err != nil {
		return tokenclassification.Response{}, err
	}
	if response.GetTokens() == nil {
		return tokenclassification.Response{}, nil
	}

	tokens := make([]tokenclassification.Token, len(response.Tokens))
	for i, token := range response.Tokens {
		tokens[i] = tokenclassification.Token{
			Text:  token.Text,
			Start: int(token.Start),
			End:   int(token.End),
			Label: token.Label,
			Score: token.Score,
		}
	}
	return tokenclassification.Response{
		Tokens: tokens,
	}, nil
}

func grpcAggregationStrategy(value tokenclassification.AggregationStrategy) tokenclassificationv1.ClassifyRequest_AggregationStrategy {
	switch value {
	case tokenclassification.AggregationStrategyNone:
		return tokenclassificationv1.ClassifyRequest_NONE
	case tokenclassification.AggregationStrategySimple:
		return tokenclassificationv1.ClassifyRequest_SIMPLE
	default:
		panic(fmt.Sprintf("client: invalid aggreagation strategy %v", value))
	}
}
