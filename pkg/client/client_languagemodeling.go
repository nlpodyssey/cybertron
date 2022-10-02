// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package client

import (
	"context"
	"fmt"
	"time"

	languagemodelingv1 "github.com/nlpodyssey/cybertron/pkg/server/gen/proto/go/languagemodeling/v1"
	"github.com/nlpodyssey/cybertron/pkg/tasks/languagemodeling"
)

var _ languagemodeling.Interface = &clientForLanguageModeling{}

// clientForLanguageModeling is a client for language modeling implementing languagemodeling.Interface
type clientForLanguageModeling struct {
	// target is the server endpoint.
	target string
	// opts is the gRPC options for the client.
	opts Options
}

// NewClientForLanguageModeling creates a new client for language modeling.
func NewClientForLanguageModeling(target string, opts Options) languagemodeling.Interface {
	return &clientForLanguageModeling{
		target: target,
		opts:   opts,
	}
}

// Predict predicts the words according to the language modeling architecture.
func (c *clientForLanguageModeling) Predict(ctx context.Context, text string, parameters languagemodeling.Parameters) (languagemodeling.Response, error) {
	conn, err := Dial(ctx, c.target, c.opts)
	if err != nil {
		return languagemodeling.Response{}, fmt.Errorf("failed to dial %q: %w", c.target, err)
	}
	cc := languagemodelingv1.NewLanguageModelingServiceClient(conn)

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	response, err := cc.Predict(ctx, &languagemodelingv1.LanguageModelingRequest{
		Input: text,
		Parameters: &languagemodelingv1.LanguageModelingParameters{
			K: int32(parameters.K),
		},
	})
	if err != nil {
		return languagemodeling.Response{}, err
	}
	if response.GetTokens() == nil {
		return languagemodeling.Response{}, nil
	}

	tokens := make([]languagemodeling.Token, len(response.Tokens))
	for i, token := range response.Tokens {
		tokens[i] = languagemodeling.Token{
			Start:  int(token.Start),
			End:    int(token.End),
			Words:  token.Words,
			Scores: token.Scores,
		}
	}
	return languagemodeling.Response{
		Tokens: tokens,
	}, nil
}
