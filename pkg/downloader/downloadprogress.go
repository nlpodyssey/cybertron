// Copyright 2022 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package downloader

import (
	"fmt"
	"sync"
	"time"

	"github.com/rs/zerolog/log"
)

// downloadProgress is a helper struct for reporting download progress.
type downloadProgress struct {
	contentLength     int
	readContentLength int
	stopCh            chan struct{}
	wg                sync.WaitGroup
}

const downloadProgressUpdateFrequency = 3 * time.Second

func newDownloadProgress(contentLength int) *downloadProgress {
	return &downloadProgress{
		contentLength:     contentLength,
		readContentLength: 0,
		stopCh:            nil,
	}
}

// Start starts the progress reporting goroutine.
func (dp *downloadProgress) Start() {
	dp.stopCh = make(chan struct{}, 1)
	dp.wg.Add(1)
	go dp.goRoutine()
}

// Stop stops the progress reporting goroutine.
func (dp *downloadProgress) Stop() {
	dp.stopCh <- struct{}{}
	dp.wg.Wait()
	dp.stopCh = nil
}

func (dp *downloadProgress) goRoutine() {
	stopCh := dp.stopCh

	for {
		select {
		case <-stopCh:
			dp.reportProgress()
			close(stopCh)
			dp.wg.Done()
			return
		case <-time.After(downloadProgressUpdateFrequency):
			dp.reportProgress()
		}
	}
}

func (dp *downloadProgress) reportProgress() {
	cl := dp.contentLength
	rcl := dp.readContentLength
	hrcl := humanizeBytesSize(rcl)

	switch {
	case cl < 0:
		log.Debug().Msgf("%s downloaded", hrcl)
	case cl == rcl:
		log.Debug().Msgf("%s (100%%) downloaded", hrcl)
	default:
		hcl := humanizeBytesSize(cl)
		perc := rcl * 100 / cl
		log.Debug().Msgf("%s of %s (%d%%) downloaded", hrcl, hcl, perc)
	}
}

// Write satisfies io.Writer interface.
func (dp *downloadProgress) Write(p []byte) (int, error) {
	dp.readContentLength += len(p)
	return len(p), nil
}

func humanizeBytesSize(n int) string {
	switch {
	case n < 1024:
		return fmt.Sprintf("%d B", n)
	case n < 1_048_576:
		return fmt.Sprintf("%.2f KiB", float64(n)/1024)
	case n < 1_073_741_824:
		return fmt.Sprintf("%.2f MiB", float64(n)/1_048_576)
	default:
		return fmt.Sprintf("%.2f GiB", float64(n)/1_073_741_824)
	}
}
