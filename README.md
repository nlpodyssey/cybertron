# Cybertron

Cybertron is a package in pure Go built upon [spaGO](https://github.com/nlpodyssey/spago) that provides Go developers with an easy interface to use NLP technologies, without needing other programming languages or complex frameworks. It's designed for using pre-trained Transformer models available on the [HuggingFace models repository](https://huggingface.co/models).

The package is primarily aimed at running **inference** with the possibility of adding fine-tuning in the future.

The team is open to contributors to expedite its growth.

## Supported models

- BERT
- ELECTRA
- BART
- PEGASUS
- MarianMT

## Supported tasks

- Masked Language Modeling
- Supervised and Zero-Shot Text Classification (Sentiment Analysis, Topic Classification, Intent Detection, ...)
- Token Classification (Named Entity Recognition, Part-of-Speech Tagging, ...)
- Extractive and Abstractive Question-Answering
- Text Encoding (Text Embedding, Semantic Search, ...)
- Text Generation (Translation, Paraphrasing, Summarization, ...)
- Relation Extraction

# Usage

Requirements:

* [Go 1.21](https://golang.org/dl/)

Clone this repo or get the library:

```console
go get -u github.com/nlpodyssey/cybertron
```

Cybertron supports two main use cases, which are explained more in detail in the following.

## Server mode

Settings are configured in a `.env` file, which is automatically loaded by Cybertron. Alternatively, it also accepts configurations via flags.

For a complete list run:

```console
GOARCH=amd64 go run ./cmd/server -h
```

Output:

```console
Usage of server:
  -address value
        server listening address
  -allowed-origins value
        allowed origins (comma separated)
  -loglevel value
        zerolog global level
  -model value
        model name (and sub-path of models-dir)
  -model-conversion value
        model conversion policy ("always"|"missing"|"never")
  -model-conversion-precision value
        floating-point bits of precision to use if the model is converted ("32"|"64")
  -model-download value
        model downloading policy ("always"|"missing"|"never")
  -models-dir value
        models's base directory
  -network value
        network type for server listening
  -task value
        type of inference/computation that the model can fulfill ("textgeneration"|"zero-shot-classification"|"question-answering"|"text-classification"|"token-classification"|"text-encoding")
  -tls value
        whether to enable TLS ("true"|"false")
  -tls-cert value
        TLS cert filename
  -tls-key value
        TLS key filename

```

For example, to run Cybertron in server mode for Machine Translation (e.g. `en` to `it`) with default settings, simply create a `.env` file in the current directory:

```console
echo "CYBERTRON_MODEL=Helsinki-NLP/opus-mt-en-it" > .env
echo "CYBERTRON_MODELS_DIR=models" >> .env
echo "CYBERTRON_MODEL_TASK=text-generation" >> .env
```

and execute the following command:

```console
GOARCH=amd64 go run ./cmd/server
```

To test the server, run:

```console
curl -X 'POST' \
  '0.0.0.0:8080/v1/generate' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "input": "You must be the change you wish to see in the world.",
  "parameters": {}
}'
```

## Library mode

Several examples can be leveraged to tour the current NLP capabilities in Cybertron. A list of the demos now follows.

### Machine Translation (En -> It)

```
GOARCH=amd64 go run ./examples/textgeneration
```

### Zero-Shot Text Classification

```
GOARCH=amd64 go run ./examples/zeroshotclassification politics,business,science,technology,health,culture,sports
```

# Dependencies

Cybertron's pricipal dependencies are:

- [Spago](https://github.com/nlpodyssey/spago) - a lightweight self-contained machine learning framework in pure Go
- [GoPickle](https://github.com/nlpodyssey/gopickle) - a Go module for loading Python's data serialized with pickle and PyTorch module files
- [GoTokenizers](https://github.com/nlpodyssey/gotokenizers) - Go implementation of today's most used tokenizers

The rest are mainly for gRPC and HTTP API developments.

# Dev Tools

> This section is intended for developers who want to change or enrich the Cybertron gRPC and HTTP APIs.

To get started, you need [buf](https://github.com/bufbuild/buf) installed in your machine. 

Then install the following tools:

```
go install github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-grpc-gateway \
  github.com/grpc-ecosystem/grpc-gateway/v2/protoc-gen-openapiv2 \
  google.golang.org/protobuf/cmd/protoc-gen-go \
  google.golang.org/grpc/cmd/protoc-gen-go-grpc
```

Then run the following command to generate the gRPC and HTTP APIs:

```
go generate ./...
```
