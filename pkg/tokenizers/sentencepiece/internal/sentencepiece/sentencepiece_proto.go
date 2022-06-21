// Copyright (c) 2020 Vikesh Raj C. All rights reserved.
// This work is licensed under the terms of the MIT license.
// For a copy, see <https://opensource.org/licenses/MIT>.

package sentencepiece

import (
	"fmt"
	"os"

	"github.com/nlpodyssey/gotokenizers/vocabulary"
	"google.golang.org/protobuf/proto"
)

// NewSentencepieceFromFile creates sentencepiece from file.
func NewSentencepieceFromFile(filename string, lowercase bool) (Sentencepiece, error) {
	s := NewEmptySentencepiece(lowercase)
	bytes, err := os.ReadFile(filename)
	if err != nil {
		return s, fmt.Errorf("unable to read file : %s, err %v", filename, err)
	}
	var model ModelProto
	err = proto.Unmarshal(bytes, &model)
	if err != nil {
		return s, fmt.Errorf("unable to read model file : %s, err %v", filename, err)
	}

	count := 0
	for i, piece := range model.GetPieces() {
		typ := piece.GetType()
		word := piece.GetPiece()
		switch typ {
		case ModelProto_SentencePiece_NORMAL, ModelProto_SentencePiece_USER_DEFINED:
			s.insert(word, piece.GetScore(), int32(i))
		case ModelProto_SentencePiece_UNKNOWN:
			s.SetUnknownIndex(int32(i))
		case ModelProto_SentencePiece_CONTROL:
			s.SetControlWord(word, int32(i))
		}
		count++
	}

	return s, nil
}

func NewSentencepieceAndVocabFromFile(filename string, lowercase bool) (Sentencepiece, *vocabulary.Vocabulary, error) {
	s := NewEmptySentencepiece(lowercase)
	bytes, err := os.ReadFile(filename)
	if err != nil {
		return s, nil, fmt.Errorf("unable to read file : %s, err %v", filename, err)
	}
	var model ModelProto
	err = proto.Unmarshal(bytes, &model)
	if err != nil {
		return s, nil, fmt.Errorf("unable to read model file : %s, err %v", filename, err)
	}

	vocab := vocabulary.NewVocabulary()

	for i, piece := range model.GetPieces() {
		if i == 2 {
			s.SetControlWord("<mask_1>", 2)
			vocab.AddTerm("<mask_1>")

			s.SetControlWord("<mask_2>", 3)
			vocab.AddTerm("<mask_2>")

			for j := int32(2); j < 103; j++ {
				unk := fmt.Sprintf("<unk_%d>", j)
				//fmt.Println(j+2, unk)
				s.SetControlWord(unk, j+2)
				vocab.AddTerm(unk)
			}
		}
		if i >= 2 {
			i += 103
		}

		typ := piece.GetType()
		word := piece.GetPiece()

		switch typ {
		case ModelProto_SentencePiece_NORMAL, ModelProto_SentencePiece_USER_DEFINED:
			s.insert(word, piece.GetScore(), int32(i))
		case ModelProto_SentencePiece_UNKNOWN:
			s.SetUnknownIndex(int32(i))
		case ModelProto_SentencePiece_CONTROL:
			s.SetControlWord(word, int32(i))
		}

		vocab.AddTerm(word)
	}

	return s, vocab, nil
}
