package collections

import (
	"fmt"

	"github.com/nlpodyssey/gopickle/types"
)

type DefaultDictClass struct{}

type DefaultDict struct {
	*types.Dict
	DefaultFactory any
}

func (DefaultDictClass) Call(args ...any) (any, error) {
	if len(args) != 1 {
		return nil, fmt.Errorf("DefaultDictClass: want 1 argument, got %d: %#v", len(args), args)
	}

	return &DefaultDict{
		Dict:           types.NewDict(),
		DefaultFactory: args[0],
	}, nil
}
