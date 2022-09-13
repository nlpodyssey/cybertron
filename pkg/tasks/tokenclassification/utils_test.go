// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tokenclassification

import (
	"reflect"
	"testing"
)

func TestAggregate(t *testing.T) {
	type testCase struct {
		input []Token
		want  []Token
	}

	two2Two := func(li1, li2, lw1, lw2 string) testCase {
		return testCase{
			input: []Token{
				{Text: "foo", Start: 0, End: 3, Label: li1},
				{Text: "bar", Start: 4, End: 7, Label: li2},
			},
			want: []Token{
				{Text: "foo", Start: 0, End: 3, Label: lw1},
				{Text: "bar", Start: 4, End: 7, Label: lw2},
			},
		}
	}

	two2One := func(li1, li2, lw string) testCase {
		return testCase{
			input: []Token{
				{Text: "foo", Start: 0, End: 3, Label: li1},
				{Text: "bar", Start: 4, End: 7, Label: li2},
			},
			want: []Token{
				{Text: "foo bar", Start: 0, End: 7, Label: lw},
			},
		}
	}

	tests := []testCase{
		// IOB
		{
			input: []Token{
				{Text: "foo", Start: 0, End: 3, Label: "O"},
				{Text: "Bar", Start: 4, End: 7, Label: "B-b"},
				{Text: "Baz", Start: 8, End: 11, Label: "I-c"},
				{Text: "qux", Start: 12, End: 15, Label: "O"},
			},
			want: []Token{
				{Text: "foo", Start: 0, End: 3, Label: ""},
				{Text: "Bar Baz", Start: 4, End: 11, Label: "b"},
				{Text: "qux", Start: 12, End: 15, Label: ""},
			},
		},

		// BIOES
		{
			input: []Token{
				{Text: "foo", Start: 0, End: 3, Label: "O"},
				{Text: "Bar", Start: 4, End: 7, Label: "S-b"},
				{Text: "baz", Start: 8, End: 11, Label: "O"},
			},
			want: []Token{
				{Text: "foo", Start: 0, End: 3, Label: ""},
				{Text: "Bar", Start: 4, End: 7, Label: "b"},
				{Text: "baz", Start: 8, End: 11, Label: ""},
			},
		},
		{
			input: []Token{
				{Text: "foo", Start: 0, End: 3, Label: "O"},
				{Text: "Bar", Start: 4, End: 7, Label: "B-b"},
				{Text: "Baz", Start: 8, End: 11, Label: "E-c"},
				{Text: "qux", Start: 12, End: 15, Label: "O"},
			},
			want: []Token{
				{Text: "foo", Start: 0, End: 3, Label: ""},
				{Text: "Bar Baz", Start: 4, End: 11, Label: "b"},
				{Text: "qux", Start: 12, End: 15, Label: ""},
			},
		},
		{
			input: []Token{
				{Text: "foo", Start: 0, End: 3, Label: "O"},
				{Text: "Bar", Start: 4, End: 7, Label: "B-b"},
				{Text: "The", Start: 8, End: 11, Label: "I-c"},
				{Text: "Baz", Start: 12, End: 15, Label: "E-d"},
				{Text: "qux", Start: 16, End: 19, Label: "O"},
			},
			want: []Token{
				{Text: "foo", Start: 0, End: 3, Label: ""},
				{Text: "Bar The Baz", Start: 4, End: 15, Label: "b"},
				{Text: "qux", Start: 16, End: 19, Label: ""},
			},
		},

		// BILOU
		{
			input: []Token{
				{Text: "foo", Start: 0, End: 3, Label: "O"},
				{Text: "Bar", Start: 4, End: 7, Label: "U-b"},
				{Text: "baz", Start: 8, End: 11, Label: "O"},
			},
			want: []Token{
				{Text: "foo", Start: 0, End: 3, Label: ""},
				{Text: "Bar", Start: 4, End: 7, Label: "b"},
				{Text: "baz", Start: 8, End: 11, Label: ""},
			},
		},
		{
			input: []Token{
				{Text: "foo", Start: 0, End: 3, Label: "O"},
				{Text: "Bar", Start: 4, End: 7, Label: "B-b"},
				{Text: "Baz", Start: 8, End: 11, Label: "L-c"},
				{Text: "qux", Start: 12, End: 15, Label: "O"},
			},
			want: []Token{
				{Text: "foo", Start: 0, End: 3, Label: ""},
				{Text: "Bar Baz", Start: 4, End: 11, Label: "b"},
				{Text: "qux", Start: 12, End: 15, Label: ""},
			},
		},
		{
			input: []Token{
				{Text: "foo", Start: 0, End: 3, Label: "O"},
				{Text: "Bar", Start: 4, End: 7, Label: "B-b"},
				{Text: "The", Start: 8, End: 11, Label: "I-c"},
				{Text: "Baz", Start: 12, End: 15, Label: "L-d"},
				{Text: "qux", Start: 16, End: 19, Label: "O"},
			},
			want: []Token{
				{Text: "foo", Start: 0, End: 3, Label: ""},
				{Text: "Bar The Baz", Start: 4, End: 15, Label: "b"},
				{Text: "qux", Start: 16, End: 19, Label: ""},
			},
		},

		// Common pairs for BIO, BIOES, BILOU

		two2Two("B-a", "B-b", "a", "b"),
		two2One("B-a", "I-b", "a"),
		two2Two("B-a", "O", "a", ""),

		two2Two("I-a", "B-b", "a", "b"),
		two2One("I-a", "I-b", "a"),
		two2Two("I-a", "O", "a", ""),

		two2Two("O", "B-b", "", "b"),
		two2Two("O", "I-b", "", "b"),
		two2Two("O", "O", "", ""),

		// Pairs for BIOES

		two2One("B-a", "E-b", "a"),
		two2Two("B-a", "S-b", "a", "b"),

		two2One("I-a", "E-b", "a"),
		two2Two("I-a", "S-b", "a", "b"),

		two2Two("O", "E-b", "", "b"),
		two2Two("O", "S-b", "", "b"),

		two2Two("E-a", "B-b", "a", "b"),
		two2Two("E-a", "I-b", "a", "b"),
		two2Two("E-a", "O", "a", ""),
		two2Two("E-a", "E-b", "a", "b"),
		two2Two("E-a", "S-b", "a", "b"),

		two2Two("S-a", "B-b", "a", "b"),
		two2Two("S-a", "I-b", "a", "b"),
		two2Two("S-a", "O", "a", ""),
		two2Two("S-a", "E-b", "a", "b"),
		two2Two("S-a", "S-b", "a", "b"),

		// Pairs for BILOU

		two2One("B-a", "L-b", "a"),
		two2Two("B-a", "U-b", "a", "b"),

		two2One("I-a", "L-b", "a"),
		two2Two("I-a", "U-b", "a", "b"),

		two2Two("L-a", "B-b", "a", "b"),
		two2Two("L-a", "I-b", "a", "b"),
		two2Two("L-a", "O", "a", ""),
		two2Two("L-a", "L-b", "a", "b"),
		two2Two("L-a", "U-b", "a", "b"),

		two2Two("O", "L-b", "", "b"),
		two2Two("O", "U-b", "", "b"),

		two2Two("U-a", "B-b", "a", "b"),
		two2Two("U-a", "I-b", "a", "b"),
		two2Two("U-a", "L-b", "a", "b"),
		two2Two("U-a", "O", "a", ""),
		two2Two("U-a", "U-b", "a", "b"),

		// Invalid labels

		two2Two("xx", "yy", "xx", "yy"),
		two2Two("", "", "", ""),

		two2Two("B-a", "yy", "a", "yy"),
		two2Two("I-a", "yy", "a", "yy"),
		two2Two("O", "yy", "", "yy"),
		two2Two("E-a", "yy", "a", "yy"),
		two2Two("S-a", "yy", "a", "yy"),
		two2Two("L-a", "yy", "a", "yy"),
		two2Two("U-a", "yy", "a", "yy"),

		two2Two("xx", "B-b", "xx", "b"),
		two2Two("xx", "I-b", "xx", "b"),
		two2Two("xx", "O", "xx", ""),
		two2Two("xx", "E-b", "xx", "b"),
		two2Two("xx", "S-b", "xx", "b"),
		two2Two("xx", "L-b", "xx", "b"),
		two2Two("xx", "U-b", "xx", "b"),

		two2Two("B-a", "", "a", ""),
		two2Two("I-a", "", "a", ""),
		two2Two("O", "", "", ""),
		two2Two("E-a", "", "a", ""),
		two2Two("S-a", "", "a", ""),
		two2Two("L-a", "", "a", ""),
		two2Two("U-a", "", "a", ""),

		two2Two("", "B-b", "", "b"),
		two2Two("", "I-b", "", "b"),
		two2Two("", "O", "", ""),
		two2Two("", "E-b", "", "b"),
		two2Two("", "S-b", "", "b"),
		two2Two("", "L-b", "", "b"),
		two2Two("", "U-b", "", "b"),
	}

	for _, tt := range tests {
		got := Aggregate(tt.input)
		if !reflect.DeepEqual(tt.want, got) {
			t.Errorf(
				"input: %v\nwant: %v\ngot:  %v",
				tt.input, tt.want, got,
			)
		}
	}
}
