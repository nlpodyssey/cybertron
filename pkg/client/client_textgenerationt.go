// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package client

import (
	"context"
	"fmt"
	"time"

	textgenerationv1 "github.com/nlpodyssey/cybertron/pkg/server/gen/proto/go/textgeneration/v1"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textgeneration"
	"github.com/nlpodyssey/cybertron/pkg/utils/nullable"
)

var _ textgeneration.Interface = &clientForTextGeneration{}

// clientForTextGeneration is a client for text generation implementing textgeneration.Interface
type clientForTextGeneration struct {
	// target is the server endpoint.
	target string
	// opts is the gRPC options for the client.
	opts Options
}

// NewClientForTextGeneration creates a new client for text generation.
func NewClientForTextGeneration(target string, opts Options) textgeneration.Interface {
	return &clientForTextGeneration{
		target: target,
		opts:   opts,
	}
}

// Generate generates text (e.g. translation, summarization, paraphrase) from the given input.
func (c *clientForTextGeneration) Generate(ctx context.Context, text string, opts *textgeneration.Options) (textgeneration.Response, error) {
	if opts == nil {
		opts = textgeneration.DefaultOptions()
	}
	topK64 := nullable.Type[int64]{
		Value: int64(opts.TopK.Value),
		Valid: opts.TopK.Valid,
	}

	conn, err := Dial(ctx, c.target, c.opts)
	if err != nil {
		return textgeneration.Response{}, fmt.Errorf("failed to dial %q: %w", c.target, err)
	}
	cc := textgenerationv1.NewTextGenerationServiceClient(conn)

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	response, err := cc.Generate(ctx, &textgenerationv1.GenerateRequest{
		Input: text,
		Parameters: &textgenerationv1.TextGenerationParameters{
			Temperature: opts.Temperature.ValuePtr(),
			DoSample:    opts.Sample.ValuePtr(),
			TopK:        topK64.ValuePtr(),
			TopP:        opts.TopP.ValuePtr(),
		},
	})
	if err != nil {
		return textgeneration.Response{}, err
	}
	return textgeneration.Response{
		Texts:  response.Texts,
		Scores: response.Scores,
	}, nil
}
