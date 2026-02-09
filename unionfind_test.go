package hdbscan

import "testing"

func TestNewUnionFind(t *testing.T) {
	uf := NewUnionFind(5)

	// Each element should be its own root.
	for i := 0; i < 5; i++ {
		if root := uf.Find(i); root != i {
			t.Errorf("Find(%d) = %d, want %d", i, root, i)
		}
	}

	// Each element has size 1.
	for i := 0; i < 5; i++ {
		if uf.size[i] != 1 {
			t.Errorf("size[%d] = %d, want 1", i, uf.size[i])
		}
	}
}

func TestUnionFind_UnionTwoElements(t *testing.T) {
	uf := NewUnionFind(5)
	root := uf.Union(1, 3)

	// Both should resolve to the same root.
	if uf.Find(1) != uf.Find(3) {
		t.Error("after Union(1,3), Find(1) != Find(3)")
	}
	// Root should be one of them.
	if root != uf.Find(1) {
		t.Errorf("Union returned %d, but Find(1) = %d", root, uf.Find(1))
	}
	// Size of the root should be 2.
	if uf.size[root] != 2 {
		t.Errorf("size of root = %d, want 2", uf.size[root])
	}
}

func TestUnionFind_MultipleUnions(t *testing.T) {
	uf := NewUnionFind(6)

	// Union {0,1,2} and {3,4,5}.
	uf.Union(0, 1)
	uf.Union(1, 2)
	uf.Union(3, 4)
	uf.Union(4, 5)

	// Same component.
	if uf.Find(0) != uf.Find(2) {
		t.Error("0 and 2 should be in same set")
	}
	if uf.Find(3) != uf.Find(5) {
		t.Error("3 and 5 should be in same set")
	}
	// Different components.
	if uf.Find(0) == uf.Find(3) {
		t.Error("0 and 3 should be in different sets")
	}

	// Union the two components.
	uf.Union(2, 4)

	// All should be connected now.
	root := uf.Find(0)
	for i := 1; i < 6; i++ {
		if uf.Find(i) != root {
			t.Errorf("after full union, Find(%d) != Find(0)", i)
		}
	}
	if uf.size[root] != 6 {
		t.Errorf("size of root = %d, want 6", uf.size[root])
	}
}

func TestUnionFind_PathCompression(t *testing.T) {
	uf := NewUnionFind(5)

	// Create a chain: 0←1←2←3←4
	uf.Union(0, 1) // root of {0,1}
	r01 := uf.Find(0)
	uf.Union(r01, 2)
	r012 := uf.Find(0)
	uf.Union(r012, 3)
	r0123 := uf.Find(0)
	uf.Union(r0123, 4)

	// Find(4) should compress path.
	root := uf.Find(4)
	// After compression, parent[4] should point directly to root.
	if uf.parent[4] != root {
		t.Errorf("after Find(4), parent[4] = %d, want root %d", uf.parent[4], root)
	}
}

func TestUnionFind_UnionBySize(t *testing.T) {
	uf := NewUnionFind(4)

	// Union {0,1,2} → size 3.
	uf.Union(0, 1)
	uf.Union(0, 2)

	bigRoot := uf.Find(0)

	// Union with single element 3 → smaller attaches to larger.
	uf.Union(3, 0)
	newRoot := uf.Find(3)

	if newRoot != bigRoot {
		t.Errorf("expected union-by-size: small tree attaches to big root %d, got root %d", bigRoot, newRoot)
	}
}
