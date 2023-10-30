// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"context"

	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	textgenerationv1 "github.com/nlpodyssey/cybertron/pkg/server/gen/proto/go/textgeneration/v1"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textgeneration"
	"github.com/nlpodyssey/cybertron/pkg/utils/nullable"
	"google.golang.org/grpc"
)

// serverForTextGeneration is a server that provides gRPC and HTTP/2 APIs for Interface task.
type serverForTextGeneration struct {
	textgenerationv1.UnimplementedTextGenerationServiceServer
	generator textgeneration.Interface
}

func NewServerForTextGeneration(generator textgeneration.Interface) RequestHandler {
	return &serverForTextGeneration{generator: generator}
}

func (s *serverForTextGeneration) RegisterServer(r grpc.ServiceRegistrar) error {
	textgenerationv1.RegisterTextGenerationServiceServer(r, s)
	return nil
}

func (s *serverForTextGeneration) RegisterHandlerServer(ctx context.Context, mux *runtime.ServeMux) error {
	return textgenerationv1.RegisterTextGenerationServiceHandlerServer(ctx, mux, s)
}

// Generate handles the Generate request.
func (s *serverForTextGeneration) Generate(ctx context.Context, req *textgenerationv1.GenerateRequest) (*textgenerationv1.GenerateResponse, error) {
	opts := req.GetParameters()
	if opts == nil {
		opts = &textgenerationv1.TextGenerationParameters{}
	}
	result, err := s.generator.Generate(ctx, req.GetInput(), &textgeneration.Options{
		Temperature: nullable.Any(opts.Temperature),
		Sample:      nullable.Any(opts.DoSample),
		TopK:        nullable.Int(opts.TopK),
		TopP:        nullable.Any(opts.TopP),
	})
	if err != nil {
		return nil, err
	}
	resp := &textgenerationv1.GenerateResponse{
		Texts:  result.Texts,
		Scores: result.Scores,
	}
	return resp, nil
}
