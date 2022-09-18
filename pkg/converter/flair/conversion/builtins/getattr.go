package builtins

import (
	"fmt"

	"github.com/nlpodyssey/cybertron/pkg/converter/flair/conversion"
)

type Getattr struct{}

func (Getattr) Call(args ...any) (any, error) {
	if len(args) != 2 && len(args) != 3 {
		return nil, fmt.Errorf("builtins.getattr: want 2 or 3 args, got %d: %#v", len(args), args)
	}

	object, ok := args[0].(conversion.PyAttributeGettable)
	if !ok {
		return nil, fmt.Errorf("builtins.getattr: 1st arg (object) does not satisfy PyAttributeGettable interface: %T", args[0])
	}

	name, ok := args[1].(string)
	if !ok {
		return nil, fmt.Errorf("builtins.getattr: want 2nd arg (name) to be string, got %T: %#v", args[1], args[1])
	}

	value, exists, err := object.PyGetAttribute(name)
	if err != nil {
		return nil, fmt.Errorf("builtins.getattr(%#v): PyGetAttribute failed: %w", args, err)
	}

	if len(args) == 3 && !exists {
		return args[2], nil
	}
	return value, nil
}
