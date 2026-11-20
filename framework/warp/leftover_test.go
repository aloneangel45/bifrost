package warp

import (
	"context"
	"errors"
	"testing"

	"github.com/maximhq/bifrost/core/schemas"
	"github.com/stretchr/testify/require"
)

// leftoverTestService is a Warp on vectors, with shared as the shared store.
func leftoverTestService(t *testing.T, vectors, shared *fakeWarpVectorStore, source VectorStoreSource) *Service {
	t.Helper()
	service := NewService(nil, WithConfigStore(&recordingStore{row: validWarpConfigRow()}), WithVectorStore(vectors),
		WithVectorStoreSource(source), WithSharedVectorStore(shared))
	t.Cleanup(service.Shutdown)
	return service
}

// sharedWithOldIndex is a shared store holding the index Warp built there
// before it moved, marked as that store's.
func sharedWithOldIndex(t *testing.T) *fakeWarpVectorStore {
	t.Helper()
	shared := newFakeWarpVectorStore()
	_, err := ensureWarpNamespace(context.Background(), shared, schemas.WarpDefaultLogVectorStoreNamespace, 1536, nil)
	require.NoError(t, err)
	shared.existing = []string{schemas.WarpDefaultLogVectorStoreNamespace}
	return shared
}

// On a store of its own, once Warp has written to it, Warp finds the index it
// left in the shared store, deletes it on request, and then finds none. Before
// it has written there, it offers nothing, since it cannot yet tell its own
// store from the shared one.
func TestWarpFindsAndDeletesItsIndexLeftInTheSharedStore(t *testing.T) {
	ctx := context.Background()
	shared := sharedWithOldIndex(t)
	service := leftoverTestService(t, newFakeWarpVectorStore(), shared, VectorStoreSource{Dedicated: true, Type: "qdrant"})

	leftover, err := service.SharedStoreLeftover(ctx)
	require.NoError(t, err)
	require.Nil(t, leftover, "nothing before Warp has marked its own store")
	_, err = service.storeID(ctx) // Warp writes to its own store
	require.NoError(t, err)

	leftover, err = service.SharedStoreLeftover(ctx)
	require.NoError(t, err)
	require.Equal(t, &LeftoverIndex{Namespace: schemas.WarpDefaultLogVectorStoreNamespace}, leftover)

	require.NoError(t, service.DeleteSharedStoreLeftover(ctx))
	require.Equal(t, []string{schemas.WarpDefaultLogVectorStoreNamespace}, shared.deleted)

	shared.existing = nil // the namespace is gone
	leftover, err = service.SharedStoreLeftover(ctx)
	require.NoError(t, err)
	require.Nil(t, leftover)
	require.ErrorIs(t, service.DeleteSharedStoreLeftover(ctx), ErrNoLeftoverIndex)
}

// Nothing is left behind while Warp uses the shared store, or when the shared
// store never held a Warp namespace.
func TestWarpFindsNoLeftoverWithoutOne(t *testing.T) {
	ctx := context.Background()
	shared := sharedWithOldIndex(t)
	onShared := leftoverTestService(t, shared, shared, VectorStoreSource{})
	leftover, err := onShared.SharedStoreLeftover(ctx)
	require.NoError(t, err)
	require.Nil(t, leftover, "the shared store's index is the live one")
	require.ErrorIs(t, onShared.DeleteSharedStoreLeftover(ctx), ErrNoLeftoverIndex)

	neverIndexed := leftoverTestService(t, newFakeWarpVectorStore(), newFakeWarpVectorStore(), VectorStoreSource{Dedicated: true, Type: "qdrant"})
	leftover, err = neverIndexed.SharedStoreLeftover(ctx)
	require.NoError(t, err)
	require.Nil(t, leftover)
}

// warp.vector_store written as a different store but reaching the shared one
// (another spelling of the same host) holds Warp's live marker: it is the
// index Warp writes to now, never one to delete.
func TestWarpDoesNotOfferItsLiveIndexUnderAnotherConfiguration(t *testing.T) {
	ctx := context.Background()
	shared := sharedWithOldIndex(t)
	sameStore := leftoverTestService(t, shared, shared, VectorStoreSource{Dedicated: true, Type: "qdrant"})
	leftover, err := sameStore.SharedStoreLeftover(ctx)
	require.NoError(t, err)
	require.Nil(t, leftover)
	require.ErrorIs(t, sameStore.DeleteSharedStoreLeftover(ctx), ErrNoLeftoverIndex)
	require.Empty(t, shared.deleted)
}

// A shared store that cannot be listed leaves the question open: an error,
// and nothing deleted.
func TestWarpDeletesNothingWhenTheSharedStoreCannotBeAsked(t *testing.T) {
	ctx := context.Background()
	shared := sharedWithOldIndex(t)
	shared.listErr = errors.New("shared store unavailable")
	service := leftoverTestService(t, newFakeWarpVectorStore(), shared, VectorStoreSource{Dedicated: true, Type: "qdrant"})
	_, err := service.SharedStoreLeftover(ctx)
	require.ErrorContains(t, err, "shared store unavailable")
	require.ErrorContains(t, service.DeleteSharedStoreLeftover(ctx), "shared store unavailable")
	require.Empty(t, shared.deleted)
}

// An index from before markers, in the store Warp writes to now under another
// spelling in config.json, holds no marker yet. Until Warp has marked its own
// store, which store is live cannot be told, so nothing is offered for
// deletion; once it has, the marker shows it is the same store.
func TestWarpDoesNotOfferAnUnmarkedIndexBeforeItsOwnStoreIsMarked(t *testing.T) {
	ctx := context.Background()
	store := newFakeWarpVectorStore()
	require.NoError(t, store.Add(ctx, schemas.WarpDefaultLogVectorStoreNamespace, "log-1", []float32{1}, map[string]interface{}{"warp_log": true}))
	store.existing = []string{schemas.WarpDefaultLogVectorStoreNamespace}
	service := leftoverTestService(t, store, store, VectorStoreSource{Dedicated: true, Type: "qdrant"})

	leftover, err := service.SharedStoreLeftover(ctx)
	require.NoError(t, err)
	require.Nil(t, leftover, "Warp's own store has no marker yet")
	require.ErrorIs(t, service.DeleteSharedStoreLeftover(ctx), ErrNoLeftoverIndex)
	require.Empty(t, store.deleted)
	require.Zero(t, store.createCalls, "a settings read creates nothing")

	_, err = service.storeID(ctx) // Warp writes to its store and marks it
	require.NoError(t, err)
	leftover, err = service.SharedStoreLeftover(ctx)
	require.NoError(t, err)
	require.Nil(t, leftover, "the same store holds Warp's live marker")
}
