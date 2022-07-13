// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package client

import (
	"context"
	"time"

	textclassificationv1 "github.com/nlpodyssey/cybertron/pkg/server/gen/proto/go/textclassification/v1"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textclassification"
)

var _ textclassification.Interface = &clientForTextClassification{}

// clientForTextClassification is a client for text classification implementing textclassification.Interface
type clientForTextClassification struct {
	// target is the server endpoint.
	target string
	// opts is the gRPC options for the client.
	opts Options
}

// NewClientForTextClassification creates a new client for text classification.
func NewClientForTextClassification(target string, opts Options) textclassification.Interface {
	return &clientForTextClassification{
		target: target,
		opts:   opts,
	}
}

// Classify classifies the given text.
func (c *clientForTextClassification) Classify(ctx context.Context, text string) (textclassification.Response, error) {
	conn, err := Dial(ctx, c.target, c.opts)
	cc := textclassificationv1.NewTextClassificationServiceClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	response, err := cc.Classify(ctx, &textclassificationv1.ClassifyRequest{
		Input: text,
	})
	if err != nil {
		return textclassification.Response{}, err
	}
	return textclassification.Response{
		Labels: response.Labels,
		Scores: response.Scores,
	}, nil
}
