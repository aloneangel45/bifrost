package warp

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore"
	"github.com/maximhq/bifrost/framework/vectorstore"
)

// Warp tells one vector store from another by asking the store, not by reading
// its configuration. When Warp creates its log namespace it writes one marker
// entry into it, holding a random store ID, and every later question of which
// store an index or a backfill checkpoint belongs to compares that ID. The
// same store written two ways in config.json (a default port spelled out, a
// trailing slash on a path) reads back the same marker; a different store, or
// a memory-only one after a restart, has none or another. The marker carries
// no warp_log flag, and every Warp read filters on it, so search, topics,
// retention and the vector budget never see it.

// warpStoreMarkerID is the marker's entry ID: a fixed UUID, since Qdrant and
// Weaviate accept nothing else, the same in every Warp namespace.
var warpStoreMarkerID = uuid.NewSHA1(uuid.NameSpaceOID, []byte("bifrost-warp-store-marker")).String()

// warpStoreIDProperty holds the marker's store ID.
const warpStoreIDProperty = "warp_store_id"

// unmarkedStoreID names, in a backfill checkpoint, a store that would not take
// the marker. It matches a store that still has none, so the run resumes there,
// and never a marked store. Two stores that both refuse the marker cannot be
// told apart.
const unmarkedStoreID = "unmarked"

// errMarkerRefused is a store that would not take the marker's write.
var errMarkerRefused = errors.New("the vector store refused Warp's store marker")

// errNoStoreMarker is a namespace that holds no marker (yet).
var errNoStoreMarker = errors.New("warp vector namespace holds no store marker")

// readStoreMarker returns the store ID the namespace's marker holds, or
// errNoStoreMarker. A store that cannot be asked is an error of its own, never
// taken for a missing marker: a marker written over it would give the store a
// new identity.
func readStoreMarker(ctx context.Context, store vectorstore.VectorStore, namespace string) (string, error) {
	result, err := store.GetChunk(ctx, namespace, warpStoreMarkerID)
	if err != nil {
		if isNotFound(err) {
			return "", errNoStoreMarker
		}
		return "", fmt.Errorf("read Warp's store marker: %w", err)
	}
	id, _ := result.Properties[warpStoreIDProperty].(string)
	if id == "" {
		return "", errNoStoreMarker
	}
	return id, nil
}

// isNotFound reports a lookup of an entry or namespace that is not there.
// Backends say so differently: most wrap vectorstore.ErrNotFound, Pinecone
// answers "not found: <id>".
func isNotFound(err error) bool {
	return errors.Is(err, vectorstore.ErrNotFound) || strings.Contains(strings.ToLower(err.Error()), "not found")
}

// storeMarkerLock takes a lock shared by every replica for writing the marker
// in namespace, and returns its release. nil writes unlocked: one process.
type storeMarkerLock func(ctx context.Context, namespace string) (release func(), err error)

// newStoreMarkerLock builds the marker lock on the config store every replica
// shares, keyed by namespace. The lock manager needs a logger.
func newStoreMarkerLock(locks configstore.LockStore, logger schemas.Logger) storeMarkerLock {
	if locks == nil || logger == nil {
		return nil
	}
	manager := configstore.NewDistributedLockManager(locks, logger, configstore.WithDefaultTTL(30*time.Second))
	return func(ctx context.Context, namespace string) (func(), error) {
		_, release, err := manager.Acquire(ctx, "warp-store-marker:"+namespace)
		return release, err
	}
}

// markerReadBack bounds the wait for a written marker to read back.
var markerReadBack = struct {
	tries    int
	interval time.Duration
}{tries: 20, interval: 250 * time.Millisecond}

// ensureStoreMarker writes the namespace's marker unless it has one. Replicas
// that find none at once must not each write their own: a backfill could
// record the first ID before the second replaced it, and then be refused as
// another store's. So the write happens under the lock every replica shares,
// after reading again, and the lock is held until the marker reads back, since
// some stores (Pinecone) show a write only after a moment and the next holder
// would otherwise miss it.
func ensureStoreMarker(ctx context.Context, store vectorstore.VectorStore, namespace string, dimension int, lock storeMarkerLock) error {
	_, err := readStoreMarker(ctx, store, namespace)
	if !errors.Is(err, errNoStoreMarker) {
		return err
	}
	if dimension <= 0 {
		return fmt.Errorf("write Warp's store marker: no embedding dimension")
	}
	if lock != nil {
		release, err := lock(ctx, namespace)
		if err != nil {
			return fmt.Errorf("lock Warp's store marker: %w", err)
		}
		defer release()
		// Another replica may have written it while this one waited.
		if _, err := readStoreMarker(ctx, store, namespace); !errors.Is(err, errNoStoreMarker) {
			return err
		}
	}
	id := uuid.NewString()
	vector := make([]float32, dimension)
	vector[0] = 1
	if err := store.Add(ctx, namespace, warpStoreMarkerID, vector, map[string]interface{}{warpStoreIDProperty: id}); err != nil {
		return fmt.Errorf("%w: %w", errMarkerRefused, err)
	}
	for try := 0; ; try++ {
		got, err := readStoreMarker(ctx, store, namespace)
		if err == nil && got == id {
			return nil
		}
		if try == markerReadBack.tries-1 {
			return fmt.Errorf("read back Warp's store marker: not there after writing it")
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(markerReadBack.interval):
		}
	}
}

// currentStoreID is the ID on the marker in the store Warp indexes into now,
// or "" when that store has none yet. It only reads: a status poll or a
// settings read must not create a namespace in a store Warp has not written to.
func (s *Service) currentStoreID(ctx context.Context) (string, error) {
	s.mu.RLock()
	vectors := s.vectorStore
	s.mu.RUnlock()
	if vectors == nil {
		return "", s.noVectorStore()
	}
	view, err := s.ConfigView(ctx)
	if err != nil {
		return "", err
	}
	id, err := readStoreMarker(ctx, vectors, normalizedNamespace(view.LogVectorStoreNamespace))
	if errors.Is(err, errNoStoreMarker) {
		return "", nil
	}
	return id, err
}

// storeID is the ID of the store Warp indexes into now, for a backfill about to
// write to it. The namespace and its marker are created first if they are not
// there yet: a new store has neither until Warp writes to it. A store that will
// not take the marker gives unmarkedStoreID, so Warp keeps working.
func (s *Service) storeID(ctx context.Context) (string, error) {
	s.mu.RLock()
	vectors := s.vectorStore
	s.mu.RUnlock()
	if vectors == nil {
		return "", s.noVectorStore()
	}
	view, err := s.ConfigView(ctx)
	if err != nil {
		return "", err
	}
	if view.EmbeddingModel == "" || view.EmbeddingDimension <= 0 {
		return "", ErrUnavailable
	}
	namespace := normalizedNamespace(view.LogVectorStoreNamespace)
	id, err := readStoreMarker(ctx, vectors, namespace)
	if !errors.Is(err, errNoStoreMarker) {
		return id, err
	}
	if _, err := ensureWarpNamespace(ctx, vectors, namespace, view.EmbeddingDimension, s.markerLock); err != nil {
		return "", err
	}
	if err := ensureStoreMarker(ctx, vectors, namespace, view.EmbeddingDimension, s.markerLock); errors.Is(err, errMarkerRefused) {
		s.warnf("Warp's vector store will not take its store marker, so a backfill cannot tell it from another unmarked store: %v", err)
		return unmarkedStoreID, nil
	} else if err != nil {
		return "", err // the lock or the read-back: try again, never a guess
	}
	return readStoreMarker(ctx, vectors, namespace)
}
