package main

import (
	"fmt"
	"os"

	"github.com/nlpodyssey/cybertron/pkg/converter"
)

func main() {
	modelDir:=os.Args[1]
	fmt.Printf("Converting model from dir %s\n", modelDir)
	err:=converter.Convert[float32](modelDir,true)
	if err!=nil{
		panic(err)
	}
}
