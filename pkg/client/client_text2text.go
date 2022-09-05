// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package client

import (
	"context"
	"time"

	text2textv1 "github.com/nlpodyssey/cybertron/pkg/server/gen/proto/go/text2text/v1"
	"github.com/nlpodyssey/cybertron/pkg/tasks/text2text"
	"github.com/nlpodyssey/cybertron/pkg/utils/nullable"
)

var _ text2text.Interface = &clientForTextGeneration{}

// clientForTextGeneration is a client for text generation implementing text2text.Interface
type clientForTextGeneration struct {
	// target is the server endpoint.
	target string
	// opts is the gRPC options for the client.
	opts Options
}

// NewClientForTextGeneration creates a new client for text generation.
func NewClientForTextGeneration(target string, opts Options) text2text.Interface {
	return &clientForTextGeneration{
		target: target,
		opts:   opts,
	}
}

// Generate generates text (e.g. translation, summarization, paraphrase) from the given input.
func (c *clientForTextGeneration) Generate(ctx context.Context, text string, opts *text2text.Options) (text2text.Response, error) {
	if opts == nil {
		opts = text2text.DefaultOptions()
	}
	topK64 := nullable.Type[int64]{
		Value: int64(opts.TopK.Value),
		Valid: opts.TopK.Valid,
	}

	conn, err := Dial(ctx, c.target, c.opts)
	cc := text2textv1.NewText2TextServiceClient(conn)

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	response, err := cc.Generate(ctx, &text2textv1.GenerateRequest{
		Input: text,
		Parameters: &text2textv1.Text2TextParameters{
			Temperature: opts.Temperature.ValuePtr(),
			DoSample:    opts.Sample.ValuePtr(),
			TopK:        topK64.ValuePtr(),
			TopP:        opts.TopP.ValuePtr(),
		},
	})
	if err != nil {
		return text2text.Response{}, err
	}
	return text2text.Response{
		Texts:  response.Texts,
		Scores: response.Scores,
	}, nil
}
