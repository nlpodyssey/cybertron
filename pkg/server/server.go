// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package server

import (
	"context"
	"crypto/tls"
	"errors"
	"fmt"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/grpc-ecosystem/grpc-gateway/v2/runtime"
	"github.com/nlpodyssey/cybertron/pkg/tasks/languagemodeling"
	"github.com/nlpodyssey/cybertron/pkg/tasks/questionanswering"
	"github.com/nlpodyssey/cybertron/pkg/tasks/text2text"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textclassification"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textencoding"
	"github.com/nlpodyssey/cybertron/pkg/tasks/tokenclassification"
	"github.com/nlpodyssey/cybertron/pkg/tasks/zeroshotclassifier"
	"github.com/rs/cors"
	"github.com/rs/zerolog/log"
	"golang.org/x/net/http2"
	"golang.org/x/net/http2/h2c"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
)

const (
	// DefaultNetwork is the default network.
	DefaultNetwork = "tcp4"
	// DefaultAddress is the default address.
	DefaultAddress = ":8080"
)

// Server is a server that provides gRPC and HTTP/2 APIs.
type Server struct {
	conf    *Config
	handler RequestHandler
	health  *health.Server
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

// RequestHandler is implemented by any task-specific service that can be
// registered in the main Server.
type RequestHandler interface {
	RegisterServer(grpc.ServiceRegistrar) error
	RegisterHandlerServer(context.Context, *runtime.ServeMux) error
}

// ResolveRequestHandler instantiates a new task-server based on the model.
func ResolveRequestHandler(model any) (RequestHandler, error) {
	switch m := model.(type) {
	case text2text.Interface:
		return NewServerForTextGeneration(m), nil
	case zeroshotclassifier.Interface:
		return NewServerForZeroShotClassification(m), nil
	case questionanswering.Interface:
		return NewServerForQuestionAnswering(m), nil
	case textclassification.Interface:
		return NewServerForTextClassification(m), nil
	case textencoding.Interface:
		return NewServerForTextEncoding(m), nil
	case tokenclassification.Interface:
		return NewServerForTokenClassification(m), nil
	case languagemodeling.Interface:
		return NewServerForLanguageModeling(m), nil
	default:
		return nil, fmt.Errorf("failed to resolve register funcs for model/task type %T", m)
	}
}

// New creates a new server.
func New(conf *Config, handler RequestHandler) *Server {
	setBaselineConfig(conf)
	return &Server{
		conf:    conf,
		handler: handler,
		health:  health.NewServer(),
	}
}

func setBaselineConfig(c *Config) {
	if c.Network == "" {
		c.Network = DefaultNetwork
	}
	if c.Address == "" {
		c.Address = DefaultAddress
	}
}

// Start up the server and block until the context is done.
func (s *Server) Start(ctx context.Context) error {
	conf := s.conf

	grpcServer := grpc.NewServer()

	grpc_health_v1.RegisterHealthServer(grpcServer, s.health)

	if err := s.handler.RegisterServer(grpcServer); err != nil {
		return fmt.Errorf("failed to register gRPC server: %w", err)
	}

	mux := runtime.NewServeMux()
	if err := s.handler.RegisterHandlerServer(ctx, mux); err != nil {
		return fmt.Errorf("failed to register gRPC handler server: %w", err)
	}

	lis, err := net.Listen(conf.Network, conf.Address)
	if err != nil {
		return fmt.Errorf("failed to listen on %s (%s): %w", conf.Address, conf.Network, err)
	}

	if strings.HasSuffix(conf.Address, ":0") {
		conf.Address = lis.Addr().String()
	}

	handler := cors.New(s.corsOptions()).Handler(mux)
	handler = s.handlerFunc(grpcServer, handler)

	err = s.serve(ctx, lis, handler)
	if err != nil && !errors.Is(err, http.ErrServerClosed) {
		return fmt.Errorf("failed to serve: %w", err)
	}
	log.Info().Msg("server stopped serving successfully")
	return nil
}

// corsOptions returns the CORS options for the server.
func (s *Server) corsOptions() cors.Options {
	return cors.Options{
		AllowedOrigins: s.conf.AllowedOrigins,
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

func (s *Server) serve(ctx context.Context, lis net.Listener, handler http.Handler) error {
	if s.conf.TLSEnabled {
		return s.serveTLS(ctx, lis, handler)
	}
	return s.serveInsecure(ctx, lis, handler)
}

// serveTLS starts the server with TLS.
func (s *Server) serveTLS(ctx context.Context, lis net.Listener, handler http.Handler) error {
	conf := s.conf

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

	idleConnsClosed := make(chan struct{})
	go func() {
		s.shutDownServerWhenContextIsDone(ctx, hs)
		close(idleConnsClosed)
	}()

	s.health.Resume()
	err = hs.Serve(tls.NewListener(lis, hs.TLSConfig))
	<-idleConnsClosed
	return err
}

// serveInsecure starts the server without TLS.
func (s *Server) serveInsecure(ctx context.Context, lis net.Listener, handler http.Handler) error {
	conf := s.conf

	h2s := &http2.Server{}
	h1s := &http.Server{
		Handler: h2c.NewHandler(handler, h2s),
	}

	log.Info().Str("network", conf.Network).Str("address", conf.Address).Bool("TLS", conf.TLSEnabled).Msg("server listening")

	idleConnsClosed := make(chan struct{})
	go func() {
		s.shutDownServerWhenContextIsDone(ctx, h1s)
		close(idleConnsClosed)
	}()

	s.health.Resume()
	err := h1s.Serve(lis)
	<-idleConnsClosed
	return err
}

const shutdownTimeout = 10 * time.Second

// shutDownServerWhenContextIsDone shuts down the server when the context is done.
func (s *Server) shutDownServerWhenContextIsDone(ctx context.Context, hs *http.Server) {
	<-ctx.Done()
	log.Info().Msg("context done, shutting down server")
	s.health.Shutdown()

	sdCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()
	err := hs.Shutdown(sdCtx)
	if err != nil {
		log.Err(err).Msg("failed to shut down server")
	}
	log.Info().Msg("server shut down successfully")
}

// ReadyForConnections returns `true` if the server is ready to accept requests.
// If after the duration `dur` the server is still not ready, returns `false`.
func (s *Server) ReadyForConnections(dur time.Duration) bool {
	return s.readyForConnections(dur) == nil
}

func (s *Server) readyForConnections(d time.Duration) error {
	end := time.Now().Add(d)
	for time.Now().Before(end) {
		if err := s.check(); err == nil {
			return nil
		}
		if d > 25*time.Millisecond {
			time.Sleep(25 * time.Millisecond)
		}
	}
	return fmt.Errorf("failed to be ready for connections after %s", d)
}

// check checks if the server is ready for connections.
func (s *Server) check() error {
	conn, err := net.Dial(s.conf.Network, s.conf.Address)
	if err != nil {
		return fmt.Errorf("failed to connect to %s (%s): %w", s.conf.Address, s.conf.Network, err)
	}
	conn.Close()
	return nil
}

// ClientAddr returns the Address used to connect clients (without the network).
// Helpful in testing when we designate a random port (0).
func (s *Server) ClientAddr() string {
	return s.conf.Address
}
