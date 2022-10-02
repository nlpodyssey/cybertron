// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"context"

	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	langaugemodelingnv1 "github.com/nlpodyssey/cybertron/pkg/server/gen/proto/go/languagemodeling/v1"
	"github.com/nlpodyssey/cybertron/pkg/tasks/languagemodeling"
	"google.golang.org/grpc"
)

// serverForLanguageModeling is a server that provides gRPC and HTTP/2 APIs for Language Modeling task.
type serverForLanguageModeling struct {
	langaugemodelingnv1.UnimplementedLanguageModelingServiceServer
	predictor languagemodeling.Interface
}

func NewServerForLanguageModeling(predictor languagemodeling.Interface) RequestHandler {
	return &serverForLanguageModeling{predictor: predictor}
}

func (s *serverForLanguageModeling) RegisterServer(r grpc.ServiceRegistrar) error {
	langaugemodelingnv1.RegisterLanguageModelingServiceServer(r, s)
	return nil
}

func (s *serverForLanguageModeling) RegisterHandlerServer(ctx context.Context, mux *runtime.ServeMux) error {
	return langaugemodelingnv1.RegisterLanguageModelingServiceHandlerServer(ctx, mux, s)
}

// Predict handles the Predict request.
func (s *serverForLanguageModeling) Predict(ctx context.Context, req *langaugemodelingnv1.LanguageModelingRequest) (*langaugemodelingnv1.LanguageModelingResponse, error) {
	result, err := s.predictor.Predict(ctx, req.GetInput(), languagemodeling.Parameters{
		K: int(req.GetParameters().GetK()),
	})
	if err != nil {
		return nil, err
	}

	tokens := make([]*langaugemodelingnv1.Token, len(result.Tokens))
	for i, token := range result.Tokens {
		tokens[i] = &langaugemodelingnv1.Token{
			Words:  token.Words,
			Scores: token.Scores,
			Start:  int32(token.Start),
			End:    int32(token.End),
		}
	}
	resp := &langaugemodelingnv1.LanguageModelingResponse{
		Tokens: tokens,
	}
	return resp, nil
}
