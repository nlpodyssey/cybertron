// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"context"
	"fmt"

	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	tokenclassificationv1 "github.com/nlpodyssey/cybertron/pkg/server/gen/proto/go/tokenclassification/v1"
	"github.com/nlpodyssey/cybertron/pkg/tasks/tokenclassification"
	"google.golang.org/grpc"
)

// serverForTextClassification is a server that provides gRPC and HTTP/2 APIs for Token Classification task.
type serverForTokenClassification struct {
	tokenclassificationv1.UnimplementedTokenClassificationServiceServer
	classifier tokenclassification.Interface
}

func NewServerForTokenClassification(classifier tokenclassification.Interface) RequestHandler {
	return &serverForTokenClassification{classifier: classifier}
}

func (s *serverForTokenClassification) RegisterServer(r grpc.ServiceRegistrar) error {
	tokenclassificationv1.RegisterTokenClassificationServiceServer(r, s)
	return nil
}

func (s *serverForTokenClassification) RegisterHandlerServer(ctx context.Context, mux *runtime.ServeMux) error {
	return tokenclassificationv1.RegisterTokenClassificationServiceHandlerServer(ctx, mux, s)
}

// Classify handles the Classify request.
func (s *serverForTokenClassification) Classify(ctx context.Context, req *tokenclassificationv1.ClassifyRequest) (*tokenclassificationv1.ClassifyResponse, error) {
	result, err := s.classifier.Classify(ctx, req.GetInput(), tokenclassification.Parameters{
		AggregationStrategy: convAggregationStrategy(req.AggregationStrategy),
	})
	if err != nil {
		return nil, err
	}

	tokens := make([]*tokenclassificationv1.Token, len(result.Tokens))
	for i, token := range result.Tokens {
		tokens[i] = &tokenclassificationv1.Token{
			Text:  token.Text,
			Label: token.Label,
			Score: token.Score,
			Start: int32(token.Start),
			End:   int32(token.End),
		}
	}
	resp := &tokenclassificationv1.ClassifyResponse{
		Tokens: tokens,
	}
	return resp, nil
}

func convAggregationStrategy(strategy tokenclassificationv1.ClassifyRequest_AggregationStrategy) tokenclassification.AggregationStrategy {
	switch strategy {
	case tokenclassificationv1.ClassifyRequest_NONE:
		return tokenclassification.AggregationStrategyNone
	case tokenclassificationv1.ClassifyRequest_SIMPLE:
		return tokenclassification.AggregationStrategySimple
	default:
		panic(fmt.Sprintf("server: invalid aggregation strategy [%s] for token classification", strategy))
	}
}
