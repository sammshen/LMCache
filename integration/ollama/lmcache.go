// Package lmcache provides on-disk KV cache persistence for llama.cpp-based runners.
//
// All intelligence lives here — the host runner only needs to call Restore() before
// prefill and StoreAsync() after the first token. The host provides a KVCacheContext
// interface so this package has no dependency on the host's llama bindings.
package lmcache

/*
#cgo CFLAGS: -std=c11
#cgo CPPFLAGS: -I${SRCDIR}/../../csrc/llama_cpp

#include "lmcache.h"
#include "lmcache.c"
*/
import "C"

import (
	"crypto/sha256"
	"fmt"
	"log/slog"
	"os"
	"sync"
	"unsafe"
)

const (
	// AppendFlag matches LLAMA_STATE_SEQ_FLAGS_APPEND (2).
	AppendFlag uint32 = 2

	// tempSeqID is reserved for the seq_cp trick during chunk serialization.
	tempSeqID = 999

	defaultChunkSize = 256
	defaultMaxBytes  = 20 * 1024 * 1024 * 1024 // 20 GB
)

// KVCacheContext is the interface this package needs from the host runner.
// In Ollama, *llama.Context satisfies this without any adapter code.
type KVCacheContext interface {
	KvCacheSeqCp(srcSeqId int, dstSeqId int, p0 int, p1 int)
	KvCacheSeqRm(seqId int, p0 int, p1 int) bool
	StateSeqGetSize(seqId int) int
	StateSeqGetData(seqId int, dst []byte) int
	StateSeqSetDataExt(seqId int, src []byte, flags uint32) int
}

// Cache is the on-disk KV cache backed by liblmcache.
type Cache struct {
	c         *C.lmcache_ctx
	chunkSize int
}

// Options for cache creation.
type Options struct {
	ChunkSize uint32
	MaxBytes  uint64
}

// New creates a Cache. cacheDir is the storage root, modelPath is used to
// derive a model-specific namespace. Returns nil if cacheDir is empty.
func New(cacheDir, modelPath string) *Cache {
	return NewWithOptions(cacheDir, modelPath, Options{})
}

// NewWithOptions creates a Cache with custom chunk size and disk budget.
func NewWithOptions(cacheDir, modelPath string, opts Options) *Cache {
	if cacheDir == "" {
		return nil
	}

	chunkSize := opts.ChunkSize
	if chunkSize == 0 {
		chunkSize = defaultChunkSize
	}
	maxBytes := opts.MaxBytes
	if maxBytes == 0 {
		maxBytes = defaultMaxBytes
	}

	h := sha256.Sum256([]byte(modelPath))
	modelID := fmt.Sprintf("%x", h[:8])

	cDir := C.CString(cacheDir)
	defer C.free(unsafe.Pointer(cDir))
	cModel := C.CString(modelID)
	defer C.free(unsafe.Pointer(cModel))

	params := C.struct_lmcache_params{
		cache_dir:  cDir,
		chunk_size: C.uint32_t(chunkSize),
		max_bytes:  C.uint64_t(maxBytes),
		model_id:   cModel,
	}

	c := C.lmcache_init(params)
	if c == nil {
		slog.Warn("lmcache: failed to initialize", "dir", cacheDir)
		return nil
	}

	slog.Info("lmcache: initialized", "dir", cacheDir, "model_id", modelID)
	return &Cache{c: c, chunkSize: int(chunkSize)}
}

// NewFromEnv creates a Cache using OLLAMA_KV_CACHE_DIR. Returns nil if unset.
func NewFromEnv(modelPath string) *Cache {
	return New(os.Getenv("OLLAMA_KV_CACHE_DIR"), modelPath)
}

// Free releases the cache resources. Safe to call on nil.
func (c *Cache) Free() {
	if c != nil && c.c != nil {
		C.lmcache_free(c.c)
		c.c = nil
	}
}

// Restore loads cached KV state into the context before prefill.
// Returns the number of tokens restored (chunk-aligned), or 0 on miss.
func (c *Cache) Restore(ctx KVCacheContext, tokens []int32, seqID int) int {
	if c == nil || len(tokens) == 0 {
		return 0
	}

	nCached := int(C.lmcache_lookup(c.c,
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		C.uint32_t(len(tokens))))

	if nCached == 0 {
		return 0
	}

	restored := 0
	for chunkEnd := c.chunkSize; chunkEnd <= nCached; chunkEnd += c.chunkSize {
		blob := c.retrieve(tokens[:chunkEnd])
		if blob == nil {
			break
		}

		var flags uint32
		if chunkEnd > c.chunkSize {
			flags = AppendFlag
		}

		n := ctx.StateSeqSetDataExt(seqID, blob, flags)
		if n == 0 {
			slog.Warn("lmcache: restore failed", "chunk_end", chunkEnd)
			break
		}
		restored = chunkEnd
	}

	if restored > 0 {
		slog.Debug("lmcache: restored", "tokens", restored)
	}
	return restored
}

// StoreAsync stores new KV chunks to disk in a background goroutine.
// mu is the lock that serializes access to the llama context.
// nCached is how many tokens were already in cache (skip those chunks).
func (c *Cache) StoreAsync(ctx KVCacheContext, tokens []int32, seqID int, mu *sync.Mutex) {
	if c == nil || len(tokens) == 0 {
		return
	}

	// alignUp to first chunk boundary
	startChunk := ((0 + c.chunkSize - 1) / c.chunkSize) * c.chunkSize
	if startChunk >= len(tokens) {
		return
	}

	// Copy tokens so the goroutine doesn't race with the caller.
	tokensCopy := make([]int32, len(tokens))
	copy(tokensCopy, tokens)

	go func() {
		for chunkEnd := startChunk; chunkEnd <= len(tokensCopy); chunkEnd += c.chunkSize {
			if chunkEnd == 0 {
				continue
			}
			chunkStart := chunkEnd - c.chunkSize

			// Acquire lock — llama context is not thread-safe.
			mu.Lock()

			ctx.KvCacheSeqCp(seqID, tempSeqID, chunkStart, chunkEnd)

			size := ctx.StateSeqGetSize(tempSeqID)
			if size == 0 {
				ctx.KvCacheSeqRm(tempSeqID, -1, -1)
				mu.Unlock()
				continue
			}

			buf := make([]byte, size)
			ctx.StateSeqGetData(tempSeqID, buf)
			ctx.KvCacheSeqRm(tempSeqID, -1, -1)

			mu.Unlock()

			// Disk I/O outside the lock.
			C.lmcache_store(c.c,
				(*C.int32_t)(unsafe.Pointer(&tokensCopy[0])),
				C.uint32_t(chunkEnd),
				(*C.uint8_t)(unsafe.Pointer(&buf[0])),
				C.size_t(len(buf)))

			slog.Debug("lmcache: stored chunk", "start", chunkStart, "end", chunkEnd, "bytes", size)
		}
	}()
}

// Lookup returns the number of tokens from the prefix that are cached (chunk-aligned).
func (c *Cache) Lookup(tokens []int32) int {
	if c == nil || len(tokens) == 0 {
		return 0
	}
	return int(C.lmcache_lookup(c.c,
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		C.uint32_t(len(tokens))))
}

// Store writes one chunk blob to disk. tokens is the full prefix up to this chunk.
func (c *Cache) Store(tokens []int32, blob []byte) {
	if c == nil || len(tokens) == 0 || len(blob) == 0 {
		return
	}
	C.lmcache_store(c.c,
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		C.uint32_t(len(tokens)),
		(*C.uint8_t)(unsafe.Pointer(&blob[0])),
		C.size_t(len(blob)))
}

// Retrieve returns one chunk's blob, or nil on miss.
func (c *Cache) Retrieve(tokens []int32) []byte {
	return c.retrieve(tokens)
}

func (c *Cache) retrieve(tokens []int32) []byte {
	size := C.lmcache_retrieve(c.c,
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		C.uint32_t(len(tokens)),
		nil, 0)
	if size == 0 {
		return nil
	}

	buf := make([]byte, int(size))
	n := C.lmcache_retrieve(c.c,
		(*C.int32_t)(unsafe.Pointer(&tokens[0])),
		C.uint32_t(len(tokens)),
		(*C.uint8_t)(unsafe.Pointer(&buf[0])),
		C.size_t(len(buf)))
	if n == 0 {
		return nil
	}
	return buf[:int(n)]
}
