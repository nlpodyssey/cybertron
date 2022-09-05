// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package client

import (
	"context"
	"github.com/nlpodyssey/cybertron/pkg/utils/ptr"
	"time"

	questionansweringnv1 "github.com/nlpodyssey/cybertron/pkg/server/gen/proto/go/questionanswering/v1"
	"github.com/nlpodyssey/cybertron/pkg/tasks/questionanswering"
)

var _ questionanswering.Interface = &clientForQuestionAnswering{}

// clientForQuestionAnswering is a client for question-answering implementing questionanswering.Interface
type clientForQuestionAnswering struct {
	// target is the server endpoint.
	target string
	// opts is the gRPC options for the client.
	opts Options
}

// NewClientForQuestionAnswering creates a new client for extractive question-answering.
func NewClientForQuestionAnswering(target string, opts Options) questionanswering.Interface {
	return &clientForQuestionAnswering{
		target: target,
		opts:   opts,
	}
}

// Answer answers the given question.
func (c *clientForQuestionAnswering) Answer(ctx context.Context, question, passage string, opts *questionanswering.Options) (questionanswering.Response, error) {
	if opts == nil {
		opts = &questionanswering.Options{}
	}

	conn, err := Dial(ctx, c.target, c.opts)
	cc := questionansweringnv1.NewQuestionAnsweringServiceClient(conn)

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	response, err := cc.Answer(ctx, &questionansweringnv1.AnswerRequest{
		Question: question,
		Passage:  passage,
		Options: &questionansweringnv1.QuestionAnsweringOptions{
			MaxAnswers:    ptr.Of[int64](int64(opts.MaxAnswers)),
			MaxAnswersLen: ptr.Of[int64](int64(opts.MaxAnswerLength)),
			MaxCandidates: ptr.Of[int64](int64(opts.MaxCandidates)),
			MinScore:      ptr.Of[float64](opts.MinScore),
		},
	})
	if err != nil {
		return questionanswering.Response{}, err
	}

	answers := make([]questionanswering.Answer, len(response.Answers))
	for i, answer := range response.Answers {
		answers[i] = questionanswering.Answer{
			Text:  answer.Text,
			Start: int(answer.Start),
			End:   int(answer.End),
			Score: answer.Score,
		}
	}
	return questionanswering.Response{Answers: answers}, nil
}
