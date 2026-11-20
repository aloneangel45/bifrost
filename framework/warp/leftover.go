package warp

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/maximhq/bifrost/framework/vectorstore"
)

// When warp.vector_store gives Warp a store of its own, the log index Warp
// built in the shared vector store before stays there: nothing deletes it on
// the way out. The settings page offers to delete it, and works out whether
// there is one each time it asks, from the stores themselves: Warp is on its
// own store, and the shared store still holds Warp's namespace without Warp's
// live marker in it. Nothing about it is recorded, so there is no state for
// replicas to disagree on. Only the configured namespace is checked: one from
// before a rename stays until it is deleted by hand. An index left in a store
// of Warp's own that Bifrost no longer connects to cannot be seen from here
// either; the docs say to delete those there.

// ErrNoLeftoverIndex is a delete with no leftover index to delete.
var ErrNoLeftoverIndex = errors.New("no Warp log index is left in the shared vector store")

// LeftoverIndex is Warp's log index left in the shared vector store.
type LeftoverIndex struct {
	Namespace string `json:"namespace"`
}

// WithSharedVectorStore gives Warp the shared vector store, so it can find and
// delete the log index it left there after moving to a store of its own.
func WithSharedVectorStore(shared vectorstore.VectorStore) Option {
	return func(s *Service) { s.sharedVectorStore = shared }
}

// SharedStoreLeftover returns the log index Warp left in the shared store, or
// nil when there is none: Warp uses the shared store, or that store holds no
// Warp namespace, or the namespace there holds Warp's live marker, which makes
// it the store Warp writes to now under another name in config.json.
func (s *Service) SharedStoreLeftover(ctx context.Context) (*LeftoverIndex, error) {
	s.mu.RLock()
	vectors, shared := s.vectorStore, s.sharedVectorStore
	s.mu.RUnlock()
	if !s.vectorSource.Dedicated || vectors == nil || shared == nil {
		return nil, nil
	}
	view, err := s.ConfigView(ctx)
	if errors.Is(err, ErrUnavailable) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	namespace := normalizedNamespace(view.LogVectorStoreNamespace)
	namespaces, err := shared.ListNamespaces(ctx, namespace)
	if err != nil {
		return nil, fmt.Errorf("list the shared vector store's namespaces: %w", err)
	}
	if !slices.Contains(namespaces, namespace) {
		return nil, nil
	}
	// Warp's own store first, read only. Until Warp has written to it there is
	// no marker there, and then an unmarked index in the shared store could be
	// the same store under another spelling in config.json: nothing is offered
	// until Warp has marked its store, at its first indexed log, backfill or
	// save.
	liveID, err := s.currentStoreID(ctx)
	if err != nil {
		return nil, err
	}
	if liveID == "" {
		return nil, nil
	}
	sharedID, err := readStoreMarker(ctx, shared, namespace)
	switch {
	case errors.Is(err, errNoStoreMarker):
		// Unmarked, while Warp's own store is marked: another store's index.
	case err != nil:
		return nil, err
	case sharedID == liveID:
		return nil, nil // the store Warp writes to now
	}
	return &LeftoverIndex{Namespace: namespace}, nil
}

// DeleteSharedStoreLeftover deletes the log index Warp left in the shared
// store, after checking again that there is one: the stores decide, not what
// the page showed when it last asked.
func (s *Service) DeleteSharedStoreLeftover(ctx context.Context) error {
	leftover, err := s.SharedStoreLeftover(ctx)
	if err != nil {
		return err
	}
	if leftover == nil {
		return ErrNoLeftoverIndex
	}
	s.mu.RLock()
	shared := s.sharedVectorStore
	s.mu.RUnlock()
	if err := shared.DeleteNamespace(ctx, leftover.Namespace); err != nil && !errors.Is(err, vectorstore.ErrNotFound) {
		return fmt.Errorf("delete Warp's old log index %q from the shared vector store: %w", leftover.Namespace, err)
	}
	return nil
}
