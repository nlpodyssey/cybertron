FROM golang:1.18.3-alpine3.16 as Builder

WORKDIR /go/src/cybertron
COPY . .

ENV GOOS linux
ENV GOARCH amd64
ENV CGO_ENABLED 0
RUN go build \
      -ldflags="-extldflags=-static" \
      -o /go/bin/cybertron \
      ./cmd/server

FROM alpine:3.16

COPY --from=Builder /go/bin/cybertron /bin/cybertron

ENV GOOS linux
ENV GOARCH amd64
ENTRYPOINT ["/bin/cybertron"]