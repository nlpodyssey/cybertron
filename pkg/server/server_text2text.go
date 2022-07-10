// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"context"

	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	text2textv1 "github.com/nlpodyssey/cybertron/pkg/server/gen/proto/go/text2text/v1"
	"github.com/nlpodyssey/cybertron/pkg/tasks/text2text"
	"github.com/nlpodyssey/cybertron/pkg/utils/nullable"
	"google.golang.org/grpc"
)

// serverForTextGeneration is a server that provides gRPC and HTTP/2 APIs for Interface task.
type serverForTextGeneration struct {
	text2textv1.UnimplementedText2TextServiceServer
	generator text2text.Interface
}

func NewServerForTextGeneration(generator text2text.Interface) TaskServer {
	return &serverForTextGeneration{generator: generator}
}

func (s *serverForTextGeneration) RegisterServer(r grpc.ServiceRegistrar) error {
	text2textv1.RegisterText2TextServiceServer(r, s)
	return nil
}

func (s *serverForTextGeneration) RegisterHandlerServer(ctx context.Context, mux *runtime.ServeMux) error {
	return text2textv1.RegisterText2TextServiceHandlerServer(ctx, mux, s)
}

// Generate handles the Generate request.
func (s *serverForTextGeneration) Generate(_ context.Context, req *text2textv1.GenerateRequest) (*text2textv1.GenerateResponse, error) {
	params := req.GetParameters()
	if params == nil {
		params = &text2textv1.Text2TextParameters{}
	}
	opts := text2text.Options{
		Temperature: nullable.Any(params.Temperature),
		Sample:      nullable.Any(params.DoSample),
		TopK:        nullable.Int(params.TopK),
		TopP:        nullable.Any(params.TopP),
	}
	result, err := s.generator.Generate(req.GetInput(), opts)
	if err != nil {
		return nil, err
	}
	resp := &text2textv1.GenerateResponse{
		Texts:  result.Texts,
		Scores: result.Scores,
	}
	return resp, nil
}
