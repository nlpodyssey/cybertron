// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
	"testing"
)

func TestExtractTriplets(t *testing.T) {
	text := "<s><triplet> Punta Cana <subj> La Altagracia Province <obj> located in the administrative territorial entity <subj> Dominican Republic <obj> country <triplet> Higuey <subj> La Altagracia Province <obj> located in the administrative territorial entity <subj> Dominican Republic <obj> country <triplet> La Altagracia Province <subj> province <obj> instance of <subj> Dominican Republic <obj> country <triplet> province <subj> Dominican Republic <obj> country <triplet> Dominican Republic <subj> La Altagracia Province <obj> contains administrative territorial entity</s>"
	want := []Triplet{
		{Head: "Punta Cana", Type: "located in the administrative territorial entity", Tail: "La Altagracia Province"},
		{Head: "Punta Cana", Type: "country", Tail: "Dominican Republic"},
		{Head: "Higuey", Type: "located in the administrative territorial entity", Tail: "La Altagracia Province"},
		{Head: "Higuey", Type: "country", Tail: "Dominican Republic"},
		{Head: "La Altagracia Province", Type: "instance of", Tail: "province"},
		{Head: "La Altagracia Province", Type: "country", Tail: "Dominican Republic"},
		{Head: "province", Type: "country", Tail: "Dominican Republic"},
		{Head: "Dominican Republic", Type: "contains administrative territorial entity", Tail: "La Altagracia Province"},
	}
	triplets := ExtractTriplets(text)
	if !reflect.DeepEqual(triplets, want) {
		t.Errorf("expected:\n%v\nactual:\n%v", want, triplets)
	}
}
