#include "lmcache.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include <time.h>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

#define LMCACHE_MAX_INDEX_ENTRIES 65536
#define LMCACHE_INDEX_FILENAME   "index.bin"
#define LMCACHE_CONFIG_FILENAME  "config.json"
#define LMCACHE_CHUNKS_DIR       "chunks"
#define LMCACHE_DEFAULT_CHUNK    256
#define LMCACHE_DEFAULT_MAX_BYTES (20ULL * 1024 * 1024 * 1024)

// ---------------------------------------------------------------------------
// FNV-1a hash
// ---------------------------------------------------------------------------

static uint64_t lmcache_hash_prefix(const int32_t * tokens, uint32_t n_tokens) {
    uint64_t h = 14695981039346656037ULL;
    for (uint32_t i = 0; i < n_tokens; i++) {
        h ^= (uint64_t)(uint32_t)tokens[i];
        h *= 1099511628211ULL;
    }
    return h;
}

// ---------------------------------------------------------------------------
// Index entry
// ---------------------------------------------------------------------------

typedef struct {
    uint64_t key;
    uint64_t blob_size;
    uint64_t timestamp;
    char     filename[40]; // hex key + ".bin\0"
} lmcache_entry;

// ---------------------------------------------------------------------------
// Context
// ---------------------------------------------------------------------------

struct lmcache_ctx {
    char           cache_dir[4096];
    char           chunks_dir[4096];
    uint32_t       chunk_size;
    uint64_t       max_bytes;
    char           model_id[256];

    lmcache_entry  entries[LMCACHE_MAX_INDEX_ENTRIES];
    uint32_t       n_entries;
    uint64_t       total_bytes;
};

// ---------------------------------------------------------------------------
// Filesystem helpers
// ---------------------------------------------------------------------------

static int mkdirp(const char * path) {
    char tmp[4096];
    snprintf(tmp, sizeof(tmp), "%s", path);
    for (char * p = tmp + 1; *p; p++) {
        if (*p == '/') {
            *p = '\0';
            mkdir(tmp, 0755);
            *p = '/';
        }
    }
    return mkdir(tmp, 0755);
}

static void key_to_filename(uint64_t key, char * out, size_t out_size) {
    snprintf(out, out_size, "%016llx.bin", (unsigned long long)key);
}

static void build_chunk_path(const lmcache_ctx * ctx, const char * filename, char * out, size_t out_size) {
    snprintf(out, out_size, "%s/%s", ctx->chunks_dir, filename);
}

// ---------------------------------------------------------------------------
// Index persistence
// ---------------------------------------------------------------------------

static void index_path(const lmcache_ctx * ctx, char * out, size_t out_size) {
    snprintf(out, out_size, "%s/%s", ctx->cache_dir, LMCACHE_INDEX_FILENAME);
}

static void lmcache_save_index(lmcache_ctx * ctx) {
    char path[4096];
    index_path(ctx, path, sizeof(path));

    FILE * f = fopen(path, "wb");
    if (!f) return;

    uint32_t version = 1;
    fwrite(&version, sizeof(version), 1, f);
    fwrite(&ctx->n_entries, sizeof(ctx->n_entries), 1, f);
    fwrite(ctx->entries, sizeof(lmcache_entry), ctx->n_entries, f);
    fclose(f);
}

static void lmcache_load_index(lmcache_ctx * ctx) {
    char path[4096];
    index_path(ctx, path, sizeof(path));

    FILE * f = fopen(path, "rb");
    if (!f) {
        ctx->n_entries = 0;
        ctx->total_bytes = 0;
        return;
    }

    uint32_t version = 0;
    if (fread(&version, sizeof(version), 1, f) != 1 || version != 1) {
        fclose(f);
        ctx->n_entries = 0;
        ctx->total_bytes = 0;
        return;
    }

    uint32_t count = 0;
    if (fread(&count, sizeof(count), 1, f) != 1) {
        fclose(f);
        ctx->n_entries = 0;
        ctx->total_bytes = 0;
        return;
    }

    if (count > LMCACHE_MAX_INDEX_ENTRIES) {
        count = LMCACHE_MAX_INDEX_ENTRIES;
    }

    if (fread(ctx->entries, sizeof(lmcache_entry), count, f) != count) {
        fclose(f);
        ctx->n_entries = 0;
        ctx->total_bytes = 0;
        return;
    }

    ctx->n_entries = count;
    ctx->total_bytes = 0;
    for (uint32_t i = 0; i < count; i++) {
        ctx->total_bytes += ctx->entries[i].blob_size;
    }

    fclose(f);
}

// ---------------------------------------------------------------------------
// Config persistence (for cache invalidation on model change)
// ---------------------------------------------------------------------------

static void config_path(const lmcache_ctx * ctx, char * out, size_t out_size) {
    snprintf(out, out_size, "%s/%s", ctx->cache_dir, LMCACHE_CONFIG_FILENAME);
}

static int lmcache_check_config(lmcache_ctx * ctx) {
    char path[4096];
    config_path(ctx, path, sizeof(path));

    FILE * f = fopen(path, "r");
    if (!f) return 0; // no config yet, will create

    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';

    // Simple check: does it contain our model_id and chunk_size?
    if (strstr(buf, ctx->model_id) == NULL) return -1;

    char chunk_str[32];
    snprintf(chunk_str, sizeof(chunk_str), "%u", ctx->chunk_size);
    if (strstr(buf, chunk_str) == NULL) return -1;

    return 0;
}

static void lmcache_write_config(lmcache_ctx * ctx) {
    char path[4096];
    config_path(ctx, path, sizeof(path));

    FILE * f = fopen(path, "w");
    if (!f) return;

    fprintf(f, "{\"model_id\":\"%s\",\"chunk_size\":%u,\"version\":1}\n",
            ctx->model_id, ctx->chunk_size);
    fclose(f);
}

// ---------------------------------------------------------------------------
// LRU eviction
// ---------------------------------------------------------------------------

static void lmcache_evict_lru(lmcache_ctx * ctx) {
    while (ctx->total_bytes > ctx->max_bytes && ctx->n_entries > 0) {
        // Find entry with oldest timestamp
        uint32_t oldest_idx = 0;
        uint64_t oldest_ts  = ctx->entries[0].timestamp;
        for (uint32_t i = 1; i < ctx->n_entries; i++) {
            if (ctx->entries[i].timestamp < oldest_ts) {
                oldest_ts  = ctx->entries[i].timestamp;
                oldest_idx = i;
            }
        }

        // Delete chunk file
        char chunk_path[4096];
        build_chunk_path(ctx, ctx->entries[oldest_idx].filename, chunk_path, sizeof(chunk_path));
        unlink(chunk_path);

        ctx->total_bytes -= ctx->entries[oldest_idx].blob_size;

        // Swap with last entry and shrink
        ctx->entries[oldest_idx] = ctx->entries[ctx->n_entries - 1];
        ctx->n_entries--;
    }
}

// ---------------------------------------------------------------------------
// Index lookup by key
// ---------------------------------------------------------------------------

static int lmcache_find_entry(const lmcache_ctx * ctx, uint64_t key, uint32_t * idx_out) {
    for (uint32_t i = 0; i < ctx->n_entries; i++) {
        if (ctx->entries[i].key == key) {
            if (idx_out) *idx_out = i;
            return 1;
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Clear all cached data
// ---------------------------------------------------------------------------

static void lmcache_clear(lmcache_ctx * ctx) {
    for (uint32_t i = 0; i < ctx->n_entries; i++) {
        char chunk_path[4096];
        build_chunk_path(ctx, ctx->entries[i].filename, chunk_path, sizeof(chunk_path));
        unlink(chunk_path);
    }
    ctx->n_entries = 0;
    ctx->total_bytes = 0;

    char path[4096];
    index_path(ctx, path, sizeof(path));
    unlink(path);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

lmcache_ctx * lmcache_init(struct lmcache_params params) {
    if (!params.cache_dir || !params.model_id) {
        return NULL;
    }

    lmcache_ctx * ctx = (lmcache_ctx *)calloc(1, sizeof(lmcache_ctx));
    if (!ctx) return NULL;

    snprintf(ctx->cache_dir, sizeof(ctx->cache_dir), "%s/%s", params.cache_dir, params.model_id);
    snprintf(ctx->chunks_dir, sizeof(ctx->chunks_dir), "%s/%s", ctx->cache_dir, LMCACHE_CHUNKS_DIR);
    snprintf(ctx->model_id, sizeof(ctx->model_id), "%s", params.model_id);

    ctx->chunk_size = params.chunk_size > 0 ? params.chunk_size : LMCACHE_DEFAULT_CHUNK;
    ctx->max_bytes  = params.max_bytes  > 0 ? params.max_bytes  : LMCACHE_DEFAULT_MAX_BYTES;

    // Create directories
    mkdirp(ctx->chunks_dir);

    // Check config for model/chunk_size mismatch
    if (lmcache_check_config(ctx) != 0) {
        // Mismatch -- clear stale cache
        lmcache_clear(ctx);
    }

    lmcache_write_config(ctx);
    lmcache_load_index(ctx);

    return ctx;
}

void lmcache_free(lmcache_ctx * ctx) {
    if (!ctx) return;
    lmcache_save_index(ctx);
    free(ctx);
}

uint32_t lmcache_lookup(lmcache_ctx * ctx,
                         const int32_t * tokens, uint32_t n_tokens) {
    if (!ctx || !tokens || n_tokens == 0) return 0;

    uint32_t best = 0;
    for (uint32_t boundary = ctx->chunk_size; boundary <= n_tokens; boundary += ctx->chunk_size) {
        uint64_t key = lmcache_hash_prefix(tokens, boundary);
        if (!lmcache_find_entry(ctx, key, NULL)) {
            break;
        }
        best = boundary;
    }
    return best;
}

void lmcache_store(lmcache_ctx * ctx,
                    const int32_t * tokens, uint32_t n_tokens,
                    const uint8_t * blob, size_t blob_size) {
    if (!ctx || !tokens || !blob || blob_size == 0) return;

    uint64_t key = lmcache_hash_prefix(tokens, n_tokens);

    // Idempotent -- skip if already stored
    if (lmcache_find_entry(ctx, key, NULL)) return;

    // Evict if over budget
    ctx->total_bytes += blob_size; // temporarily add
    if (ctx->total_bytes > ctx->max_bytes) {
        lmcache_evict_lru(ctx);
    }

    // Bounds check
    if (ctx->n_entries >= LMCACHE_MAX_INDEX_ENTRIES) {
        ctx->total_bytes -= blob_size;
        return;
    }

    // Write blob to disk
    char filename[40];
    key_to_filename(key, filename, sizeof(filename));

    char chunk_path[4096];
    build_chunk_path(ctx, filename, chunk_path, sizeof(chunk_path));

    FILE * f = fopen(chunk_path, "wb");
    if (!f) {
        ctx->total_bytes -= blob_size;
        return;
    }

    if (fwrite(blob, 1, blob_size, f) != blob_size) {
        fclose(f);
        unlink(chunk_path);
        ctx->total_bytes -= blob_size;
        return;
    }
    fclose(f);

    // Add index entry
    lmcache_entry * e = &ctx->entries[ctx->n_entries++];
    e->key       = key;
    e->blob_size = blob_size;
    e->timestamp = (uint64_t)time(NULL);
    snprintf(e->filename, sizeof(e->filename), "%s", filename);

    lmcache_save_index(ctx);
}

size_t lmcache_retrieve(lmcache_ctx * ctx,
                          const int32_t * tokens, uint32_t n_tokens,
                          uint8_t * dst, size_t dst_size) {
    if (!ctx || !tokens || n_tokens == 0) return 0;

    uint64_t key = lmcache_hash_prefix(tokens, n_tokens);

    uint32_t idx;
    if (!lmcache_find_entry(ctx, key, &idx)) return 0;

    lmcache_entry * e = &ctx->entries[idx];

    // Size query
    if (!dst) return (size_t)e->blob_size;

    if (dst_size < (size_t)e->blob_size) return 0;

    char chunk_path[4096];
    build_chunk_path(ctx, e->filename, chunk_path, sizeof(chunk_path));

    FILE * f = fopen(chunk_path, "rb");
    if (!f) return 0;

    size_t n = fread(dst, 1, (size_t)e->blob_size, f);
    fclose(f);

    if (n != (size_t)e->blob_size) return 0;

    // Update LRU timestamp
    e->timestamp = (uint64_t)time(NULL);

    return n;
}
