// Copyright 2022 The NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"fmt"
	"os"
	"runtime"
	"time"

	//lint:ignore ST1001 allow dot import just to make the example more readable
	. "github.com/nlpodyssey/cybertron/examples"
	"github.com/nlpodyssey/cybertron/pkg/tasks"
	"github.com/nlpodyssey/cybertron/pkg/tasks/textgeneration"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"github.com/shirou/gopsutil/v3/cpu"
	"github.com/shirou/gopsutil/v3/mem"
	"github.com/shirou/gopsutil/v3/process"
)

func main() {
	zerolog.SetGlobalLevel(zerolog.DebugLevel)
	LoadDotenv()

	modelsDir := HasEnvVarOr("CYBERTRON_MODELS_DIR", "models")
	modelName := HasEnvVarOr("CYBERTRON_MODEL", textgeneration.DefaultModelForMachineTranslation("en", "it"))

	start := time.Now()
	m, err := tasks.Load[textgeneration.Interface](&tasks.Config{ModelsDir: modelsDir, ModelName: modelName})
	if err != nil {
		log.Fatal().Err(err).Send()
	}

	log.Debug().Msgf("Loaded model %q in %v", modelName, time.Since(start))

	logMetrics()

	opts := textgeneration.DefaultOptions()

	fn := func(text string) error {
		start := time.Now()
		result, err := m.Generate(context.Background(), text, opts)
		if err != nil {
			return err
		}
		fmt.Println(time.Since(start).Seconds())
		fmt.Println(result.Texts[0])
		runtime.GC()
		return nil
	}

	err = ForEachInput(os.Stdin, fn)
	if err != nil {
		log.Fatal().Err(err).Send()
	}
}

func logMetrics() error {
	zerolog.TimeFieldFormat = zerolog.TimeFormatUnix
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	// Get total CPU count
	totalCpu, err := cpu.Counts(false)
	if err != nil {
		return err
	}
	// Get process CPU percentage
	p, err := process.NewProcess(int32(os.Getpid()))
	if err != nil {
		return err
	}
	percent, err := p.CPUPercent()
	if err != nil {
		return err
	}

	// Log CPU Metrics
	log.Info().
		Int("total_cpu_cores", totalCpu).
		Float64("process_cpu_usage_percent", percent).
		Msg("CPU Metrics")

	// Get total available RAM
	vmStat, err := mem.VirtualMemory()
	if err != nil {
		return err
	}
	// Get process RAM usage
	memInfo, err := p.MemoryInfo()
	if err != nil {
		return err
	}

	// Log RAM Metrics
	log.Info().
		Float64("total_ram_available_mb", byteToMb(vmStat.Total)).
		Float64("process_ram_usage_mb", byteToMb(memInfo.RSS)).
		Msg("RAM Metrics")

	return nil
}

func byteToMb(b uint64) float64 {
	return float64(b) / 1024 / 1024
}
