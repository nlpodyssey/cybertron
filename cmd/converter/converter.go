package main

import (
	"fmt"
	"os"
	"strconv"

	"github.com/nlpodyssey/cybertron/pkg/converter"
)

func main() {
	if len(os.Args) != 3 {
		fmt.Printf("Usage: %s <model_path> <overwrite_if_exists>\n", os.Args[0])
		os.Exit(1)
	}
	modelDir := os.Args[1]
	overwriteIfExists, err := strconv.ParseBool(os.Args[2])
	if err != nil {
		fmt.Printf("Failed to parse overwrite_if_exists: %s\n", err)
		os.Exit(1)
	}
	fmt.Printf("Converting model from dir %s\n", modelDir)
	err = converter.Convert[float32](modelDir, overwriteIfExists)
	if err != nil {
		panic(err)
	}
}
