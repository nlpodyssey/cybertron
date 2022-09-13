// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tokenclassification

import "fmt"

func FilterNotEntities(tokens []Token) []Token {
	ret := make([]Token, 0)
	for _, token := range tokens {
		if token.Label[0] == 'O' { // outside
			continue
		}
		ret = append(ret, token)
	}
	return ret
}

func Aggregate(tokens []Token) []Token {
	a := &aggregator{
		tokens: make([]Token, 0),
	}
	for _, t := range tokens {
		a.add(t)
	}
	return a.tokens
}

type aggregator struct {
	last   byte
	tokens []Token
}

func (a *aggregator) add(t Token) {
	prefix := extractPrefix(t.Label)

	if a.canAggregate(prefix) {
		a.aggregate(t)
	} else {
		a.append(t)
	}

	a.last = prefix
}

func (a *aggregator) canAggregate(prefix byte) bool {
	return anyOf2Bytes(a.last, 'B', 'I') &&
		anyOf3Bytes(prefix, 'I', 'E', 'L')
}

func (a *aggregator) aggregate(t Token) {
	last := &a.tokens[len(a.tokens)-1]
	last.End = t.End
	last.Text = fmt.Sprintf("%s %s", last.Text, t.Text)
}

func (a *aggregator) append(t Token) {
	t.Label = stripPrefix(t.Label)
	a.tokens = append(a.tokens, t)
}

func stripPrefix(label string) string {
	switch {
	case len(label) > 2:
		return label[2:]
	default:
		return label
	}
}

func extractPrefix(label string) byte {
	if len(label) > 2 {
		return label[0]
	}
	return 0
}

func anyOf2Bytes(p, a, b byte) bool {
	return p == a || p == b
}

func anyOf3Bytes(p, a, b, c byte) bool {
	return p == a || p == b || p == c
}
