// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strings"
)

type Triplet struct {
	Head string
	Type string
	Tail string
}

func ExtractTriplets(text string) []Triplet {
	i := interpreter{curToken: otherToken}
	return i.processText(text)
}

type tokenType byte

const (
	otherToken tokenType = iota
	tripletToken
	subjectToken
	objectToken
)

type interpreter struct {
	triplets   []Triplet
	curTriplet Triplet
	curToken   tokenType
}

func (r *interpreter) processText(text string) []Triplet {
	text = r.preprocessText(text)

	for _, token := range r.splitText(text) {
		if len(token) == 0 {
			continue
		}
		r.processToken(token)
	}

	if !r.currentTripletIsBlank() {
		r.commitCurrentTriplet()
	}

	return r.triplets
}

func (r *interpreter) processToken(token string) {
	switch token {
	case "<triplet>":
		r.processTripletTag()
	case "<subj>":
		r.processSubjectTag()
	case "<obj>":
		r.processObjectTag()
	default:
		r.processTextToken(token)
	}
}

func (r *interpreter) processTripletTag() {
	r.curToken = tripletToken
	if r.curTriplet.Type != "" {
		r.commitCurrentTriplet()
		r.curTriplet.Type = ""
	}
	r.curTriplet.Head = ""
}

func (r *interpreter) processSubjectTag() {
	r.curToken = subjectToken
	if r.curTriplet.Type != "" {
		r.commitCurrentTriplet()
	}
	r.curTriplet.Tail = ""
}

func (r *interpreter) processObjectTag() {
	r.curToken = objectToken
	r.curTriplet.Type = ""
}

func (r *interpreter) processTextToken(text string) {
	switch r.curToken {
	case tripletToken:
		r.curTriplet.Head = r.concatText(r.curTriplet.Head, text)
	case subjectToken:
		r.curTriplet.Tail = r.concatText(r.curTriplet.Tail, text)
	case objectToken:
		r.curTriplet.Type = r.concatText(r.curTriplet.Type, text)
	}
}

func (r *interpreter) commitCurrentTriplet() {
	ct := r.curTriplet
	r.triplets = append(r.triplets, Triplet{
		Head: strings.TrimSpace(ct.Head),
		Type: strings.TrimSpace(ct.Type),
		Tail: strings.TrimSpace(ct.Tail),
	})
}

func (r *interpreter) currentTripletIsBlank() bool {
	ct := r.curTriplet
	return ct.Head == "" && ct.Type == "" && ct.Tail == ""
}

func (r *interpreter) preprocessText(text string) string {
	text = strings.TrimSpace(text)
	text = strings.ReplaceAll(text, "<s>", "")
	text = strings.ReplaceAll(text, "</s>", "")
	text = strings.ReplaceAll(text, "<pad>", "")
	return text
}

func (r *interpreter) splitText(text string) []string {
	return strings.Split(text, " ")
}

func (r *interpreter) concatText(a, b string) string {
	return fmt.Sprintf("%s %s", a, b)
}
