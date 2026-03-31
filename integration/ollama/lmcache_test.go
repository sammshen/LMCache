package lmcache

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNewAndFree(t *testing.T) {
	dir := t.TempDir()
	c := New(dir, "test-model")
	if c == nil {
		t.Fatal("expected non-nil cache")
	}
	c.Free()
}

func TestNewFromEnvEmpty(t *testing.T) {
	os.Unsetenv("OLLAMA_KV_CACHE_DIR")
	c := NewFromEnv("test-model")
	if c != nil {
		t.Fatal("expected nil when env is unset")
	}
}

func TestStoreAndRetrieve(t *testing.T) {
	dir := t.TempDir()
	c := NewWithOptions(dir, "test-model", Options{ChunkSize: 4})
	if c == nil {
		t.Fatal("expected non-nil cache")
	}
	defer c.Free()

	tokens := []int32{1, 2, 3, 4, 5, 6, 7, 8}
	blob1 := []byte("chunk-one-data-here")
	blob2 := []byte("chunk-two-data-here")

	c.Store(tokens[:4], blob1)
	c.Store(tokens[:8], blob2)

	if got := c.Retrieve(tokens[:4]); string(got) != string(blob1) {
		t.Errorf("chunk 1: expected %q, got %q", blob1, got)
	}
	if got := c.Retrieve(tokens[:8]); string(got) != string(blob2) {
		t.Errorf("chunk 2: expected %q, got %q", blob2, got)
	}
}

func TestLookup(t *testing.T) {
	dir := t.TempDir()
	c := NewWithOptions(dir, "test-model", Options{ChunkSize: 4})
	if c == nil {
		t.Fatal("expected non-nil cache")
	}
	defer c.Free()

	tokens := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

	if n := c.Lookup(tokens); n != 0 {
		t.Errorf("expected 0, got %d", n)
	}

	c.Store(tokens[:4], []byte("b1"))
	if n := c.Lookup(tokens); n != 4 {
		t.Errorf("expected 4, got %d", n)
	}

	c.Store(tokens[:8], []byte("b2"))
	if n := c.Lookup(tokens); n != 8 {
		t.Errorf("expected 8, got %d", n)
	}

	c.Store(tokens[:12], []byte("b3"))
	if n := c.Lookup(tokens); n != 12 {
		t.Errorf("expected 12, got %d", n)
	}
}

func TestLookupStopsAtGap(t *testing.T) {
	dir := t.TempDir()
	c := NewWithOptions(dir, "test-model", Options{ChunkSize: 4})
	if c == nil {
		t.Fatal("expected non-nil cache")
	}
	defer c.Free()

	tokens := []int32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
	c.Store(tokens[:4], []byte("b1"))
	c.Store(tokens[:12], []byte("b3")) // skip chunk 2

	if n := c.Lookup(tokens); n != 4 {
		t.Errorf("expected 4 (gap at chunk 2), got %d", n)
	}
}

func TestPersistence(t *testing.T) {
	dir := t.TempDir()
	tokens := []int32{1, 2, 3, 4}
	blob := []byte("persistent-data")

	c1 := NewWithOptions(dir, "test-model", Options{ChunkSize: 4})
	c1.Store(tokens, blob)
	c1.Free()

	c2 := NewWithOptions(dir, "test-model", Options{ChunkSize: 4})
	defer c2.Free()

	if got := c2.Retrieve(tokens); got == nil || string(got) != string(blob) {
		t.Errorf("expected %q after reload, got %q", blob, got)
	}
}

func TestModelMismatchClearsCache(t *testing.T) {
	dir := t.TempDir()
	tokens := []int32{1, 2, 3, 4}

	c1 := NewWithOptions(dir, "model-a", Options{ChunkSize: 4})
	c1.Store(tokens, []byte("old"))
	c1.Free()

	c2 := NewWithOptions(dir, "model-b", Options{ChunkSize: 4})
	defer c2.Free()

	if got := c2.Retrieve(tokens); got != nil {
		t.Error("expected nil after model mismatch")
	}
}

func TestRetrieveMiss(t *testing.T) {
	dir := t.TempDir()
	c := NewWithOptions(dir, "test-model", Options{ChunkSize: 4})
	defer c.Free()

	if got := c.Retrieve([]int32{1, 2, 3, 4}); got != nil {
		t.Error("expected nil for cache miss")
	}
}

func TestLRUEviction(t *testing.T) {
	dir := t.TempDir()
	c := NewWithOptions(dir, "test-model", Options{ChunkSize: 4, MaxBytes: 100})
	if c == nil {
		t.Fatal("expected non-nil cache")
	}
	defer c.Free()

	bigBlob := make([]byte, 60)
	tokens1 := []int32{1, 2, 3, 4}
	tokens2 := []int32{5, 6, 7, 8}
	tokens3 := []int32{9, 10, 11, 12}

	c.Store(tokens1, bigBlob)
	c.Store(tokens2, bigBlob)
	c.Store(tokens3, bigBlob) // triggers eviction

	if got := c.Retrieve(tokens1); got != nil {
		t.Error("expected tokens1 evicted")
	}
	if got := c.Retrieve(tokens3); got == nil {
		t.Error("expected tokens3 present")
	}

	// model ID is a hash, find the directory dynamically
	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		if e.IsDir() {
			chunks, _ := os.ReadDir(filepath.Join(dir, e.Name(), "chunks"))
			if len(chunks) > 2 {
				t.Errorf("expected <=2 chunk files, got %d", len(chunks))
			}
		}
	}
}
