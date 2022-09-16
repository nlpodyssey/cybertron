package torch

import "github.com/nlpodyssey/gopickle/types"

type RNNUnserializableClass struct{}
type RNNUnserializable struct{}

var _ types.PyNewable = RNNUnserializableClass{}

func (c RNNUnserializableClass) PyNew(args ...interface{}) (interface{}, error) {
	return &RNNUnserializable{}, nil
}
