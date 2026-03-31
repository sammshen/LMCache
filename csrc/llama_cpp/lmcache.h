#ifndef LMCACHE_H
#define LMCACHE_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct lmcache_ctx lmcache_ctx;

struct lmcache_params {
    const char * cache_dir;       // e.g., ~/.cache/lmcache/
    uint32_t     chunk_size;      // tokens per chunk (default 256)
    uint64_t     max_bytes;       // disk budget (default 20 GB)
    const char * model_id;        // model fingerprint -- prevents cross-model cache use
};

// Initialize LMCache context. Returns NULL on failure.
lmcache_ctx * lmcache_init(struct lmcache_params params);

// Free LMCache context and all associated resources.
void lmcache_free(lmcache_ctx * ctx);

// How many tokens from this prefix are cached on disk?
// Returns largest chunk-aligned count.
// O(n/chunk_size) hash probes, no disk I/O beyond the in-memory index.
uint32_t lmcache_lookup(lmcache_ctx * ctx,
                         const int32_t * tokens, uint32_t n_tokens);

// Store one chunk's opaque blob.
// `tokens[0..n_tokens)` is the full prefix up to this chunk boundary.
// The blob is written to disk as-is -- never parsed or interpreted.
void lmcache_store(lmcache_ctx * ctx,
                    const int32_t * tokens, uint32_t n_tokens,
                    const uint8_t * blob, size_t blob_size);

// Retrieve one chunk's opaque blob.
// `tokens[0..n_tokens)` identifies which chunk to retrieve.
// Returns blob size written to dst, or 0 on miss.
// Call with dst=NULL to get required size.
size_t lmcache_retrieve(lmcache_ctx * ctx,
                          const int32_t * tokens, uint32_t n_tokens,
                          uint8_t * dst, size_t dst_size);

#ifdef __cplusplus
}
#endif

#endif // LMCACHE_H
