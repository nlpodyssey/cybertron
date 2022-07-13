// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package client

import (
	"context"
	"time"

	zeroshottextclassificationv1 "github.com/nlpodyssey/cybertron/pkg/server/gen/proto/go/zeroshot/v1"
	"github.com/nlpodyssey/cybertron/pkg/tasks/zeroshotclassifier"
)

var _ zeroshotclassifier.Interface = &clientForZeroShotClassification{}

// clientForZeroShotClassification is a client for zero-shot text generation implementing zeroshotclassifier.Interface
type clientForZeroShotClassification struct {
	// target is the server endpoint.
	target string
	// opts is the gRPC options for the client.
	opts Options
}

// NewClientForZeroShotClassification creates a new client for zero-shot text classification.
func NewClientForZeroShotClassification(target string, opts Options) zeroshotclassifier.Interface {
	return &clientForZeroShotClassification{
		target: target,
		opts:   opts,
	}
}

// Classify classifies the given text.
func (c *clientForZeroShotClassification) Classify(ctx context.Context, text string, parameters zeroshotclassifier.Parameters) (zeroshotclassifier.Response, error) {
	conn, err := Dial(ctx, c.target, c.opts)
	cc := zeroshottextclassificationv1.NewZeroShotServiceClient(conn)

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	response, err := cc.Classify(ctx, &zeroshottextclassificationv1.ClassifyRequest{
		Input: text,
		Parameters: &zeroshottextclassificationv1.ZeroShotParameters{
			HypothesisTemplate: parameters.HypothesisTemplate,
			CandidateLabels:    parameters.CandidateLabels,
			MultiLabel:         parameters.MultiLabel,
		},
	})
	if err != nil {
		return zeroshotclassifier.Response{}, err
	}
	return zeroshotclassifier.Response{
		Labels: response.Labels,
		Scores: response.Scores,
	}, nil
}
