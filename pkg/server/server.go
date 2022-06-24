// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"context"
	"crypto/tls"
	"fmt"
	"net"
	"net/http"
	"strings"

	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	"github.com/nlpodyssey/cybertron/pkg/tasks/questionanswering"
	"github.com/nlpodyssey/cybertron/pkg/tasks/text2text"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textclassification"
	"github.com/nlpodyssey/cybertron/pkg/tasks/zeroshotclassifier"
	"github.com/rs/cors"
	"github.com/rs/zerolog/log"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
	"google.golang.org/grpc"
)

// Server is a server that provides gRPC and HTTP/2 APIs.
type Server struct {
	Config        Config
	RegisterFuncs RegisterFuncs
}

// Config is the configuration for the server.
type Config struct {
	Network        string
	Address        string
	AllowedOrigins []string
	TLSEnabled     bool
	TLSCert        string
	TLSKey         string
}

// DefaultServerConfig returns the default server config.
func DefaultServerConfig() Config {
	return Config{
		Network:        "tcp4",
		Address:        ":8080",
		AllowedOrigins: make([]string, 0),
		TLSEnabled:     false,
		TLSCert:        "",
		TLSKey:         "",
	}
}

// RegisterFuncs contains the gRPC and HTTP/2 handlers.
type RegisterFuncs struct {
	RegisterServer        func(grpc.ServiceRegistrar) error
	RegisterHandlerServer func(ctx context.Context, mux *runtime.ServeMux) error
}

// New creates a new server.
func New(conf Config, model any) (*Server, error) {
	regFuncs, err := resolveRegisterFuncs(model)
	if err != nil {
		return nil, fmt.Errorf("failed to resolve register funcs: %w", err)
	}
	return &Server{
		Config:        conf,
		RegisterFuncs: *regFuncs,
	}, nil
}

// resolveRegisterFuncs resolves the register funcs for the server based on the model.
func resolveRegisterFuncs(model any) (*RegisterFuncs, error) {
	switch m := model.(type) {
	case text2text.Interface:
		return registerText2TextFunc(m)
	case zeroshotclassifier.Interface:
		return registerZeroShotClassifierFunc(m)
	case questionanswering.Interface:
		return registerQuestionAnsweringFunc(m)
	case textclassification.Interface:
		return registerTextClassificationFunc(m)
	default:
		return nil, fmt.Errorf("cannot create a server for model/task type %T", m)
	}
}

// Run starts the server.
func (s *Server) Run(ctx context.Context) error {
	conf := s.Config

	grpcServer := grpc.NewServer()
	if err := s.RegisterFuncs.RegisterServer(grpcServer); err != nil {
		return fmt.Errorf("failed to register gRPC server: %w", err)
	}

	gwMux := runtime.NewServeMux()
	if err := s.RegisterFuncs.RegisterHandlerServer(ctx, gwMux); err != nil {
		return fmt.Errorf("failed to register gRPC handler server: %w", err)
	}

	mux := gwMux

	lis, err := net.Listen(conf.Network, conf.Address)
	if err != nil {
		return fmt.Errorf("failed to listen on %s (%s): %w", conf.Address, conf.Network, err)
	}

	handler := cors.New(s.corsOptions()).Handler(mux)
	handler = s.handlerFunc(grpcServer, handler)

	if conf.TLSEnabled {
		return s.serveTLS(ctx, lis, handler)
	}
	return s.serveInsecure(ctx, lis, handler)
}

// corsOptions returns the CORS options for the server.
func (s *Server) corsOptions() cors.Options {
	return cors.Options{
		AllowedOrigins: s.Config.AllowedOrigins,
	}
}

// handlerFunc returns a handler that adds the gRPC server to the HTTP/2 server.
func (s *Server) handlerFunc(grpcServer *grpc.Server, httpHandler http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if isGRPCRequest(r) {
			grpcServer.ServeHTTP(w, r)
		} else {
			httpHandler.ServeHTTP(w, r)
		}
	})
}

// isGRPCRequest returns true if the request is a gRPC request.
func isGRPCRequest(r *http.Request) bool {
	return r.ProtoMajor == 2 &&
		strings.Contains(r.Header.Get("Content-Type"), "application/grpc")
}

// serveTLS starts the server with TLS.
func (s *Server) serveTLS(ctx context.Context, lis net.Listener, handler http.Handler) error {
	conf := s.Config

	tlsCert, err := tls.LoadX509KeyPair(conf.TLSCert, conf.TLSKey)
	if err != nil {
		return fmt.Errorf("failed to load TLS public/private key pair: %w", err)
	}

	hs := &http.Server{
		Handler: handler,
		TLSConfig: &tls.Config{
			Certificates: []tls.Certificate{tlsCert},
			NextProtos:   []string{"h2"},
		},
	}

	log.Info().Str("network", conf.Network).Str("address", conf.Address).Bool("TLS", conf.TLSEnabled).Msg("server listening")

	go shutDownServerWhenContextIsDone(ctx, hs)

	err = hs.Serve(tls.NewListener(lis, hs.TLSConfig))
	if err != nil {
		return fmt.Errorf("server error: %w", err)
	}
	return nil
}

// serveInsecure starts the server without TLS.
func (s *Server) serveInsecure(ctx context.Context, lis net.Listener, handler http.Handler) error {
	conf := s.Config

	h2s := &http2.Server{}
	h1s := &http.Server{
		Handler: h2c.NewHandler(handler, h2s),
	}

	log.Info().Str("network", conf.Network).Str("address", conf.Address).Bool("TLS", conf.TLSEnabled).Msg("server listening")

	go shutDownServerWhenContextIsDone(ctx, h1s)

	err := h1s.Serve(lis)
	if err != nil {
		return fmt.Errorf("server error: %w", err)
	}
	return nil
}

// shutDownServerWhenContextIsDone shuts down the server when the context is done.
func shutDownServerWhenContextIsDone(ctx context.Context, hs *http.Server) {
	<-ctx.Done()
	log.Info().Msg("context done, shutting down server")
	err := hs.Shutdown(context.Background())
	if err != nil {
		log.Err(err).Msg("server shutdown error")
	}
}
