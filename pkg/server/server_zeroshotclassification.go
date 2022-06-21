// Copyright 2022 NLP Odyssey Authors. All rights reserved.
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

// registerZeroShotClassifierFunc registers the ZeroShotClassification functions.
func registerZeroShotClassifierFunc(classifier zeroshotclassifier.Interface) (*RegisterFuncs, error) {
	s := &serverForZeroShotClassification{classifier: classifier}
	return &RegisterFuncs{
		RegisterServer: func(r grpc.ServiceRegistrar) error {
			zeroshotv1.RegisterZeroShotServiceServer(r, s)
			return nil
		},
		RegisterHandlerServer: func(ctx context.Context, mux *runtime.ServeMux) error {
			return zeroshotv1.RegisterZeroShotServiceHandlerServer(ctx, mux, s)
		},
	}, nil
}

// Classify handles the Classify request.
func (s *serverForZeroShotClassification) Classify(_ context.Context, req *zeroshotv1.ClassifyRequest) (*zeroshotv1.ClassifyResponse, error) {
	params := req.GetParameters()
	candidateLabels := params.GetCandidateLabels()
	result, err := s.classifier.Classify(req.GetInput(), zeroshotclassifier.Parameters{
		CandidateLabels:    candidateLabels,
		HypothesisTemplate: params.GetHypothesisTemplate(),
		MultiLabel:         params.GetMultiLabel(),
	})
	if err != nil {
		return nil, err
	}

	labels := make([]string, len(result.Labels))
	for i, labelIndex := range result.Labels {
		labels[i] = candidateLabels[labelIndex]
	}

	resp := &zeroshotv1.ClassifyResponse{
		Labels: labels,
		Scores: result.Scores,
	}
	return resp, nil
}
