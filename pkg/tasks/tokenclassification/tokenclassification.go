// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tokenclassification

import (
	"context"
	"errors"
)

const (
	// DefaultEnglishModel is a model for Named Entities Recognition for the English language.
	// It supports the following entities (CoNLL-2003 NER dataset):
	// LOC, MISC, ORG, PER
	DefaultEnglishModel = "dbmdz/bert-large-cased-finetuned-conll03-english"

	// DefaultEnglishModelOntonotes is a model for Named Entities Recognition for the English language.
	// It supports the following entities:
	// CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW, LOC, MONEY, NORP, ORDINAL, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART
	// Model card: https://huggingface.co/djagatiya/ner-bert-base-cased-ontonotesv5-englishv4
	DefaultEnglishModelOntonotes = "djagatiya/ner-bert-base-cased-ontonotesv5-englishv4"

	// DefaultModelMulti is a multilingual model for Named Entities Recognition supporting 9 languages:
	// de, en, es, fr, it, nl, pl, pt, ru.
	// Model card: https://huggingface.co/Babelscape/wikineural-multilingual-ner
	DefaultModelMulti = "Babelscape/wikineural-multilingual-ner"
)

type AggregationStrategy string

const (
	// AggregationStrategyNone - Every token gets classified without further aggregation.
	AggregationStrategyNone AggregationStrategy = "none"

	// AggregationStrategySimple - Entities are grouped according to the IOB annotation schema.
	AggregationStrategySimple AggregationStrategy = "simple"
)

// ErrInputSequenceTooLong means that pre-processing the input text
// produced a sequence that exceeds the maximum allowed length.
var ErrInputSequenceTooLong = errors.New("input sequence too long")

type Parameters struct {
	AggregationStrategy AggregationStrategy
}

// Interface defines the main functions for token classification task.
type Interface interface {
	// Classify returns the classification of the given example.
	Classify(ctx context.Context, text string, parameters Parameters) (Response, error)
}

// Token is a labeled text token.
type Token struct {
	Text  string
	Start int
	End   int
	Label string
	Score float64
}

// Response contains the response from token classification.
type Response struct {
	Tokens []Token
}
