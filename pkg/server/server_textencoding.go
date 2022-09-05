// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"context"

	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	textencodingv1 "github.com/nlpodyssey/cybertron/pkg/server/gen/proto/go/textencoding/v1"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"
	"google.golang.org/grpc"
)

// serverForTextClassification is a server that provides gRPC and HTTP/2 APIs for Text Classification task.
type serverForTextEncoding struct {
	textencodingv1.UnimplementedTextEncodingServiceServer
	encoder textencoding.Interface
}

func NewServerForTextEncoding(encoder textencoding.Interface) RequestHandler {
	return &serverForTextEncoding{encoder: encoder}
}

func (s *serverForTextEncoding) RegisterServer(r grpc.ServiceRegistrar) error {
	textencodingv1.RegisterTextEncodingServiceServer(r, s)
	return nil
}

func (s *serverForTextEncoding) RegisterHandlerServer(ctx context.Context, mux *runtime.ServeMux) error {
	return textencodingv1.RegisterTextEncodingServiceHandlerServer(ctx, mux, s)
}

// Encode handles the Encode request.
func (s *serverForTextEncoding) Encode(ctx context.Context, req *textencodingv1.EncodingRequest) (*textencodingv1.EncodingResponse, error) {
	result, err := s.encoder.Encode(ctx, req.GetInput(), int(req.GetPoolingStrategy()))
	if err != nil {
		return nil, err
	}
	resp := &textencodingv1.EncodingResponse{
		Vector: result.Vector.Data().F32(),
	}
	return resp, nil
}
