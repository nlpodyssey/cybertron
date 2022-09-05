// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package client

import (
	"context"
	"fmt"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/balancer/roundrobin"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
)

// loadBalancingConfig is the configuration for the round-robin load balancer.
var loadBalancingConfig = fmt.Sprintf(`{ "loadBalancingConfig": [{"%v": {}}] }`, roundrobin.Name)

// Options are the options for dialing a gRPC client.
type Options struct {
	UseTLS        bool
	CertFile      string
	UseRoundRobin bool
}

// Dial creates a client connection to the configured target, also respecting
// the given configuration.
//
// This function blocks until the underlying connection is up, within a
// timeout of 30 seconds.
func Dial(ctx context.Context, target string, opts Options) (_ *grpc.ClientConn, err error) {
	grpcOpts := []grpc.DialOption{
		grpc.WithBlock(),
	}

	creds := insecure.NewCredentials()
	if opts.UseTLS {
		creds, err = credentials.NewClientTLSFromFile(opts.CertFile, "")
		if err != nil {
			return nil, fmt.Errorf("failed to construct TLS credentials: %v", err)
		}
	}
	grpcOpts = append(grpcOpts, grpc.WithTransportCredentials(creds))

	if opts.UseRoundRobin {
		grpcOpts = append(grpcOpts, grpc.WithDefaultServiceConfig(loadBalancingConfig))
	}

	ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(ctx, target, grpcOpts...)
	if err != nil {
		return nil, fmt.Errorf("error dialing gRPC %+v: %w", opts, err)
	}
	return conn, nil
}
