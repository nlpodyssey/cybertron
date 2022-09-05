// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"context"

	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	zeroshotv1 "github.com/nlpodyssey/cybertron/pkg/server/gen/proto/go/zeroshot/v1"
	"github.com/nlpodyssey/cybertron/pkg/tasks/zeroshotclassifier"
	"google.golang.org/grpc"
)

// serverForZeroShotClassification is a server that provides gRPC and HTTP/2 APIs for Zero-Shot Classification task.
type serverForZeroShotClassification struct {
	zeroshotv1.UnimplementedZeroShotServiceServer
	classifier zeroshotclassifier.Interface
}

func NewServerForZeroShotClassification(classifier zeroshotclassifier.Interface) RequestHandler {
	return &serverForZeroShotClassification{classifier: classifier}
}

func (s *serverForZeroShotClassification) RegisterServer(r grpc.ServiceRegistrar) error {
	zeroshotv1.RegisterZeroShotServiceServer(r, s)
	return nil
}

func (s *serverForZeroShotClassification) RegisterHandlerServer(ctx context.Context, mux *runtime.ServeMux) error {
	return zeroshotv1.RegisterZeroShotServiceHandlerServer(ctx, mux, s)
}

// Classify handles the Classify request.
func (s *serverForZeroShotClassification) Classify(ctx context.Context, req *zeroshotv1.ClassifyRequest) (*zeroshotv1.ClassifyResponse, error) {
	params := req.GetParameters()
	candidateLabels := params.GetCandidateLabels()
	result, err := s.classifier.Classify(ctx, req.GetInput(), zeroshotclassifier.Parameters{
		CandidateLabels:    candidateLabels,
		HypothesisTemplate: params.GetHypothesisTemplate(),
		MultiLabel:         params.GetMultiLabel(),
	})
	if err != nil {
		return nil, err
	}

	resp := &zeroshotv1.ClassifyResponse{
		Labels: result.Labels,
		Scores: result.Scores,
	}
	return resp, nil
}
