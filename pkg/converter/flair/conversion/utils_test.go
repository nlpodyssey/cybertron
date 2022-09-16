// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package conversion

import (
	"reflect"
	"testing"

	"github.com/nlpodyssey/gopickle/types"
)

func TestAssertType(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		var v any = 42
		vt, err := AssertType[int](v)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		if vt != 42 {
			t.Errorf("want 42, got %#v", vt)
		}
	})

	t.Run("failure", func(t *testing.T) {
		var v any = 42
		_, err := AssertType[string](v)
		if err == nil {
			t.Error("want error, got nil")
		}
	})
}

func TestAssignAssertedType(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		var v any = 42
		var vt int
		err := AssignAssertedType(v, &vt)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		if vt != 42 {
			t.Errorf("want 42, got %#v", vt)
		}
	})

	t.Run("failure", func(t *testing.T) {
		var v any = 42
		var vt string
		err := AssignAssertedType(v, &vt)
		if err == nil {
			t.Error("want error, got nil")
		}
	})
}

func TestAssignOptionalAssertedType(t *testing.T) {
	t.Run("success with value", func(t *testing.T) {
		var v any = 42
		var vt *int
		err := AssignOptionalAssertedType(v, &vt)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		if vt == nil {
			t.Fatal("want pointer to 42, got nil")
		}
		if *vt != 42 {
			t.Errorf("want 42, got %#v", vt)
		}
	})

	t.Run("success with nil", func(t *testing.T) {
		var v any = nil
		var vt *int
		err := AssignOptionalAssertedType(v, &vt)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		if vt != nil {
			t.Errorf("want nil, got %#v", vt)
		}
	})

	t.Run("failure", func(t *testing.T) {
		var v any = 42
		var vt *string
		err := AssignOptionalAssertedType(v, &vt)
		if err == nil {
			t.Error("want error, got nil")
		}
	})
}

func TestDictToMap(t *testing.T) {
	t.Run("empty dict", func(t *testing.T) {
		d := types.NewDict()
		m, err := DictToMap[string, int](d)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		if len(m) != 0 {
			t.Errorf("want empty map, got %#v", m)
		}
	})

	t.Run("successful conversion", func(t *testing.T) {
		d := types.NewDict()
		d.Set("foo", 1)
		d.Set("bar", 2)

		m, err := DictToMap[string, int](d)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		want := map[string]int{
			"foo": 1,
			"bar": 2,
		}
		if !reflect.DeepEqual(m, want) {
			t.Errorf("want %#v, got %#v", want, m)
		}
	})

	t.Run("invalid key type", func(t *testing.T) {
		d := types.NewDict()
		d.Set("foo", 1)
		d.Set(9, 2)

		_, err := DictToMap[string, int](d)
		if err == nil {
			t.Error("want error, got nil")
		}
	})

	t.Run("invalid value type", func(t *testing.T) {
		d := types.NewDict()
		d.Set("foo", 1)
		d.Set("bar", "baz")

		_, err := DictToMap[string, int](d)
		if err == nil {
			t.Error("want error, got nil")
		}
	})
}

func TestAssignDictToMap(t *testing.T) {
	t.Run("empty dict", func(t *testing.T) {
		d := types.NewDict()
		var m map[string]int
		err := AssignDictToMap(d, &m)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		if len(m) != 0 {
			t.Errorf("want empty map, got %#v", m)
		}
	})

	t.Run("successful conversion", func(t *testing.T) {
		d := types.NewDict()
		d.Set("foo", 1)
		d.Set("bar", 2)

		var m map[string]int
		err := AssignDictToMap(d, &m)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		want := map[string]int{
			"foo": 1,
			"bar": 2,
		}
		if !reflect.DeepEqual(m, want) {
			t.Errorf("want %#v, got %#v", want, m)
		}
	})

	t.Run("invalid key type", func(t *testing.T) {
		d := types.NewDict()
		d.Set("foo", 1)
		d.Set(9, 2)

		var m map[string]int
		err := AssignDictToMap(d, &m)
		if err == nil {
			t.Error("want error, got nil")
		}
	})

	t.Run("invalid value type", func(t *testing.T) {
		d := types.NewDict()
		d.Set("foo", 1)
		d.Set("bar", "baz")

		var m map[string]int
		err := AssignDictToMap(d, &m)
		if err == nil {
			t.Error("want error, got nil")
		}
	})
}

func TestListToSlice(t *testing.T) {
	t.Run("empty list", func(t *testing.T) {
		l := types.NewList()
		s, err := ListToSlice[int](l)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		if len(s) != 0 {
			t.Errorf("want empty slice, got %#v", s)
		}
	})

	t.Run("successful conversion", func(t *testing.T) {
		l := types.NewList()
		l.Append(1)
		l.Append(2)

		s, err := ListToSlice[int](l)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		want := []int{1, 2}
		if !reflect.DeepEqual(s, want) {
			t.Errorf("want %#v, got %#v", want, s)
		}
	})

	t.Run("invalid item type", func(t *testing.T) {
		l := types.NewList()
		l.Append(1)
		l.Append("foo")

		_, err := ListToSlice[int](l)
		if err == nil {
			t.Error("want error, got nil")
		}
	})
}

func TestAssignListToSlice(t *testing.T) {
	t.Run("empty list", func(t *testing.T) {
		l := types.NewList()
		var s []int
		err := AssignListToSlice[int](l, &s)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		if len(s) != 0 {
			t.Errorf("want empty slice, got %#v", s)
		}
	})

	t.Run("successful conversion", func(t *testing.T) {
		l := types.NewList()
		l.Append(1)
		l.Append(2)

		var s []int
		err := AssignListToSlice[int](l, &s)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		want := []int{1, 2}
		if !reflect.DeepEqual(s, want) {
			t.Errorf("want %#v, got %#v", want, s)
		}
	})

	t.Run("invalid item type", func(t *testing.T) {
		l := types.NewList()
		l.Append(1)
		l.Append("foo")

		var s []int
		err := AssignListToSlice[int](l, &s)
		if err == nil {
			t.Error("want error, got nil")
		}
	})
}

func TestTupleToSlice(t *testing.T) {
	t.Run("empty list", func(t *testing.T) {
		tu := &types.Tuple{}
		s, err := TupleToSlice[int](tu)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		if len(s) != 0 {
			t.Errorf("want empty slice, got %#v", s)
		}
	})

	t.Run("successful conversion", func(t *testing.T) {
		tu := &types.Tuple{1, 2}
		s, err := TupleToSlice[int](tu)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		want := []int{1, 2}
		if !reflect.DeepEqual(s, want) {
			t.Errorf("want %#v, got %#v", want, s)
		}
	})

	t.Run("invalid item type", func(t *testing.T) {
		tu := &types.Tuple{1, "foo"}
		_, err := TupleToSlice[int](tu)
		if err == nil {
			t.Error("want error, got nil")
		}
	})
}

func TestAssignTupleToSlice(t *testing.T) {
	t.Run("empty list", func(t *testing.T) {
		tu := &types.Tuple{}
		var s []int
		err := AssignTupleToSlice[int](tu, &s)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		if len(s) != 0 {
			t.Errorf("want empty slice, got %#v", s)
		}
	})

	t.Run("successful conversion", func(t *testing.T) {
		tu := &types.Tuple{1, 2}
		var s []int
		err := AssignTupleToSlice[int](tu, &s)
		if err != nil {
			t.Errorf("want no error, got %v", err)
		}
		want := []int{1, 2}
		if !reflect.DeepEqual(s, want) {
			t.Errorf("want %#v, got %#v", want, s)
		}
	})

	t.Run("invalid item type", func(t *testing.T) {
		tu := &types.Tuple{1, "foo"}
		var s []int
		err := AssignTupleToSlice[int](tu, &s)
		if err == nil {
			t.Error("want error, got nil")
		}
	})
}
