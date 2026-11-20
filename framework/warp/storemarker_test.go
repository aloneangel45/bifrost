package warp

import (
	"context"
	"errors"
	"sync"
	"testing"
	"time"

	bifrost "github.com/maximhq/bifrost/core"
	"github.com/maximhq/bifrost/core/schemas"
	"github.com/maximhq/bifrost/framework/configstore/tables"
	"github.com/maximhq/bifrost/framework/vectorstore"
	"github.com/stretchr/testify/require"
)

// Creating Warp's namespace writes the store marker, and creating it again
// keeps the same one: the ID is the store's, not the call's. The marker is no
// log, so a read of Warp's logs never returns it.
func TestWarpStoreMarkerIsWrittenOnceAndIsNoLog(t *testing.T) {
	ctx := context.Background()
	vectors := newFakeWarpVectorStore()
	_, err := ensureWarpNamespace(ctx, vectors, "WarpLogs", 3, nil)
	require.NoError(t, err)
	first, err := readStoreMarker(ctx, vectors, "WarpLogs")
	require.NoError(t, err)
	require.NotEmpty(t, first)

	_, err = ensureWarpNamespace(ctx, vectors, "WarpLogs", 3, nil)
	require.NoError(t, err)
	again, err := readStoreMarker(ctx, vectors, "WarpLogs")
	require.NoError(t, err)
	require.Equal(t, first, again)

	// A real store, which applies the filter Warp's reads carry.
	real, err := vectorstore.NewVectorStore(ctx, &vectorstore.Config{Enabled: true, Type: vectorstore.VectorStoreTypeChromem, Config: vectorstore.ChromemConfig{}}, nil)
	require.NoError(t, err)
	_, err = ensureWarpNamespace(ctx, real, "WarpLogs", 3, nil)
	require.NoError(t, err)
	logs, _, err := real.GetAll(ctx, "WarpLogs", []vectorstore.Query{{Field: "warp_log", Operator: vectorstore.QueryOperatorEqual, Value: true}}, nil, nil, 100)
	require.NoError(t, err)
	require.Empty(t, logs, "the marker is not a log")
	count, err := vectorstore.Count(ctx, real, "WarpLogs", []vectorstore.Query{{Field: "warp_log", Operator: vectorstore.QueryOperatorEqual, Value: true}})
	require.NoError(t, err)
	require.Zero(t, count, "nor counted as one by the vector budget")
}

// The marker belongs to the data, so it survives a restart of a store on disk
// and is gone from a memory-only store after one: Chromem with a path keeps
// its ID, Chromem without one gets a new one.
func TestWarpStoreMarkerFollowsTheDataAcrossARestart(t *testing.T) {
	ctx := context.Background()
	open := func(path string) vectorstore.VectorStore {
		store, err := vectorstore.NewVectorStore(ctx, &vectorstore.Config{Enabled: true, Type: vectorstore.VectorStoreTypeChromem, Config: vectorstore.ChromemConfig{Path: path}}, nil)
		require.NoError(t, err)
		return store
	}
	idOf := func(store vectorstore.VectorStore) string {
		_, err := ensureWarpNamespace(ctx, store, "WarpLogs", 3, nil)
		require.NoError(t, err)
		id, err := readStoreMarker(ctx, store, "WarpLogs")
		require.NoError(t, err)
		return id
	}
	dir := t.TempDir()
	require.Equal(t, idOf(open(dir)), idOf(open(dir)), "a store on disk keeps its ID across a restart")
	require.NotEqual(t, idOf(open("")), idOf(open("")), "a memory-only store is a new store at every start")
}

// failingChunkStore cannot be asked for an entry.
type failingChunkStore struct{ *fakeWarpVectorStore }

func (failingChunkStore) GetChunk(context.Context, string, string) (vectorstore.SearchResult, error) {
	return vectorstore.SearchResult{}, errors.New("connection refused")
}

// A store that cannot be asked is not taken for one without a marker: writing
// a new marker over it would give the store a new identity.
func TestWarpStoreMarkerIsNotRewrittenWhenTheStoreCannotBeAsked(t *testing.T) {
	ctx := context.Background()
	vectors := newFakeWarpVectorStore()
	_, err := ensureWarpNamespace(ctx, vectors, "WarpLogs", 3, nil)
	require.NoError(t, err)
	before, err := readStoreMarker(ctx, vectors, "WarpLogs")
	require.NoError(t, err)

	err = ensureStoreMarker(ctx, failingChunkStore{vectors}, "WarpLogs", 3, nil)
	require.ErrorContains(t, err, "connection refused")
	after, err := readStoreMarker(ctx, vectors, "WarpLogs")
	require.NoError(t, err)
	require.Equal(t, before, after)
}

// Asking whether a checkpoint is this store's only reads: a status poll on a
// store Warp has not written to yet creates no namespace and no marker there,
// and a checkpoint naming a store cannot be this one's while it has none.
func TestWarpCheckpointMatchingCreatesNothing(t *testing.T) {
	ctx := context.Background()
	vectors := newFakeWarpVectorStore()
	service := NewService(nil, WithConfigStore(&recordingStore{row: validWarpConfigRow()}), WithVectorStore(vectors),
		WithVectorStoreSource(VectorStoreSource{Dedicated: true, Type: "qdrant"}))
	t.Cleanup(service.Shutdown)
	matches, err := service.BackfillMatchesStore(ctx, `{"vector_store":"some-other-store"}`)
	require.NoError(t, err)
	require.False(t, matches)
	require.Zero(t, vectors.createCalls, "no namespace is created by a read")
	require.Empty(t, vectors.idsIn(validWarpConfigRow().LogVectorStoreNamespace))
	_, err = readStoreMarker(ctx, vectors, validWarpConfigRow().LogVectorStoreNamespace)
	require.ErrorIs(t, err, errNoStoreMarker)
}

// markerRefusingStore refuses the marker's write, as a Weaviate class from
// before the marker does with auto-schema off.
type markerRefusingStore struct{ *fakeWarpVectorStore }

func (s markerRefusingStore) Add(ctx context.Context, namespace, id string, embedding []float32, metadata map[string]interface{}) error {
	if id == warpStoreMarkerID {
		return errors.New("no such property warp_store_id")
	}
	return s.fakeWarpVectorStore.Add(ctx, namespace, id, embedding, metadata)
}

// A store that will not take the marker leaves Warp working: the namespace is
// still provisioned, and a backfill starts and runs, on the shared store and on
// a dedicated one alike.
func TestWarpWorksWhenTheStoreRefusesTheMarker(t *testing.T) {
	for name, source := range map[string]VectorStoreSource{"shared": {}, "dedicated": {Dedicated: true, Type: "weaviate"}} {
		t.Run(name, func(t *testing.T) {
			ctx := context.Background()
			vectors := markerRefusingStore{newFakeWarpVectorStore()}
			_, err := ensureWarpNamespace(ctx, vectors, "WarpLogs", 3, nil)
			require.NoError(t, err, "provisioning does not fail on the marker")

			service := NewService(nil, WithConfigStore(&recordingStore{row: validWarpConfigRow()}), WithLogReader(&backfillLogReader{}),
				WithVectorStore(vectors), WithEmbeddingExecutor(backfillEmbeddingExecutor), WithVectorStoreSource(source))
			t.Cleanup(service.Shutdown)
			metaJSON, err := service.BuildBackfillJobMeta(ctx, time.Now().Add(-time.Hour), time.Now(), false)
			require.NoError(t, err, "a backfill starts without a marker")
			_, err = service.RunBackfillJob(ctx, tables.TableSidekiqJob{Metadata: metaJSON}, func(string) error { return nil })
			require.NoError(t, err, "and runs")
		})
	}
}

// memLockStore is the config store's lock table, in memory: what every replica
// shares.
type memLockStore struct {
	mu    sync.Mutex
	locks map[string]tables.TableDistributedLock
}

func (m *memLockStore) TryAcquireLock(_ context.Context, lock *tables.TableDistributedLock) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if held, ok := m.locks[lock.LockKey]; ok && held.ExpiresAt.After(time.Now()) {
		return false, nil
	}
	m.locks[lock.LockKey] = *lock
	return true, nil
}
func (m *memLockStore) GetLock(_ context.Context, key string) (*tables.TableDistributedLock, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if held, ok := m.locks[key]; ok {
		return &held, nil
	}
	return nil, nil
}
func (m *memLockStore) UpdateLockExpiry(_ context.Context, key, holder string, expiresAt time.Time) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if held, ok := m.locks[key]; ok && held.HolderID == holder {
		held.ExpiresAt = expiresAt
		m.locks[key] = held
	}
	return nil
}
func (m *memLockStore) ReleaseLock(_ context.Context, key, holder string) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if held, ok := m.locks[key]; ok && held.HolderID == holder {
		delete(m.locks, key)
		return true, nil
	}
	return false, nil
}
func (m *memLockStore) CleanupExpiredLocks(context.Context) (int64, error) { return 0, nil }
func (m *memLockStore) CleanupExpiredLockByKey(context.Context, string) (bool, error) {
	return false, nil
}

// racingMarkerStore holds the first two marker reads until both have been
// made, so two replicas both see no marker before either writes one, and
// counts the marker writes.
type racingMarkerStore struct {
	*fakeWarpVectorStore
	mu      sync.Mutex
	reads   int
	both    chan struct{}
	markers int
}

func (s *racingMarkerStore) GetChunk(ctx context.Context, namespace, id string) (vectorstore.SearchResult, error) {
	result, err := s.fakeWarpVectorStore.GetChunk(ctx, namespace, id)
	if id == warpStoreMarkerID {
		s.mu.Lock()
		s.reads++
		if s.reads == 2 {
			close(s.both)
		}
		wait := s.reads <= 2
		s.mu.Unlock()
		if wait {
			<-s.both
		}
	}
	return result, err
}

func (s *racingMarkerStore) Add(ctx context.Context, namespace, id string, embedding []float32, metadata map[string]interface{}) error {
	if id == warpStoreMarkerID {
		s.mu.Lock()
		s.markers++
		s.mu.Unlock()
	}
	return s.fakeWarpVectorStore.Add(ctx, namespace, id, embedding, metadata)
}

// Two replicas that both find no marker write one between them, not one each:
// otherwise a backfill could record the first ID before the second replaced
// it, and then be refused as another store's. The lock lives in the config
// store the replicas share.
func TestWarpReplicasWriteOneStoreMarkerBetweenThem(t *testing.T) {
	ctx := context.Background()
	store := &racingMarkerStore{fakeWarpVectorStore: newFakeWarpVectorStore(), both: make(chan struct{})}
	locks := &memLockStore{locks: map[string]tables.TableDistributedLock{}}
	logger := bifrost.NewDefaultLogger(schemas.LogLevelError)

	var wg sync.WaitGroup
	errs := make([]error, 2)
	for replica := range 2 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			errs[replica] = ensureStoreMarker(ctx, store, "WarpLogs", 3, newStoreMarkerLock(locks, logger))
		}()
	}
	wg.Wait()
	require.NoError(t, errs[0])
	require.NoError(t, errs[1])
	require.Equal(t, 1, store.markers, "one marker is written, so its ID never changes under a checkpoint")
}

// Only a refused write falls back to an unmarked store: a marker that cannot be
// locked is an error, so the backfill is retried rather than started under an
// ID that its own marker would contradict once written.
func TestWarpBackfillFailsWhenTheMarkerCannotBeLocked(t *testing.T) {
	ctx := context.Background()
	service := backfillServiceOn(t, newFakeWarpVectorStore(), dedicatedSource, nil)
	service.markerLock = func(context.Context, string) (func(), error) { return nil, errors.New("config store unavailable") }
	_, err := service.BuildBackfillJobMeta(ctx, time.Now().Add(-time.Hour), time.Now(), false)
	require.ErrorContains(t, err, "config store unavailable")
}
