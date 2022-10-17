// Copyright 2018 The gVisor Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package pgalloc contains the page allocator subsystem, which manages memory
// that may be mapped into application address spaces.
package pgalloc

import (
	"fmt"
	"math"
	"os"
	"time"

	"golang.org/x/sys/unix"
	"gvisor.dev/gvisor/pkg/abi/linux"
	"gvisor.dev/gvisor/pkg/atomicbitops"
	"gvisor.dev/gvisor/pkg/context"
	"gvisor.dev/gvisor/pkg/errors/linuxerr"
	"gvisor.dev/gvisor/pkg/hostarch"
	"gvisor.dev/gvisor/pkg/log"
	"gvisor.dev/gvisor/pkg/memutil"
	"gvisor.dev/gvisor/pkg/safemem"
	"gvisor.dev/gvisor/pkg/sentry/hostmm"
	"gvisor.dev/gvisor/pkg/sentry/memmap"
	"gvisor.dev/gvisor/pkg/sentry/usage"
	"gvisor.dev/gvisor/pkg/sync"
)

const pagesPerHugePage = hostarch.HugePageSize / hostarch.PageSize

// MemoryFile is a memmap.File whose pages may be allocated to arbitrary
// users.
type MemoryFile struct {
	// MemoryFile owns a single backing file. Each page in the backing file is
	// considered "committed" or "uncommitted". A page is committed if the host
	// kernel is spending resources to store its contents and uncommitted
	// otherwise. This definition includes pages that the host kernel has
	// swapped. This is intentional; it means that committed pages can only
	// become uncommitted as a result of MemoryFile's actions, such that page
	// commitment does not change even if host kernel swapping behavior changes.
	//
	// Each page in the MemoryFile is in one of the following conceptual states,
	// protected by mu:
	//
	// - Void: Pages beyond the backing file's current size cannot store data.
	// Extending the file's size transitions pages between the old and new sizes
	// from void to free. Void pages are assumed to be uncommitted.
	//
	// - Free: Free pages are immediately allocatable. Free pages are assumed to
	// be uncommitted, and implicitly zeroed. Free pages become used when they
	// are allocated.
	//
	// - Used: Used pages have been allocated and currently have a non-zero
	// reference count. Used pages may transition from uncommitted to committed,
	// but not vice versa. The content of unused pages is unknown. Used pages
	// become waste when their reference count becomes zero.
	//
	// - Waste: Waste pages have no users, but cannot be immediately reallocated
	// since their commitment state and content is unknown. Waste pages may be
	// uncommitted or committed, but cannot transition between the two.
	// MemoryFile's reclaimer goroutine transitions pages from waste to
	// reclaiming. Precommitting allocations can transition pages from waste to
	// used.
	//
	// - Reclaiming: Reclaiming pages are waste pages that the reclaimer
	// goroutine has made ineligible for recycling by precommitting allocations
	// by removal from waste-tracking. The reclaimer decommits reclaiming pages
	// without holding mu, then transitions them back to free with mu locked.
	//
	// - Sub-reclaimed: MemoryFile may provide allocations explicitly requesting
	// huge pages with hugepage-aligned ranges, expecting that the host will back
	// those ranges with huge pages. References are still counted at page
	// granularity within such ranges, and individual small pages within a huge
	// page may become waste and be reclaimed as a result. However, the
	// containing huge page is not reallocatable until all small pages within it
	// are reclaimed. In this case, the reclaimer conceptually transitions small
	// pages to sub-reclaimed rather than free. For consistency with legacy
	// behavior, sub-reclaimed pages are assumed to be uncommitted.

	mu memoryFileMutex

	// unwasteSmall and unwasteHuge track waste ranges backed by small/huge pages
	// respectively. Both sets are "inverted"; segments exist for all ranges that
	// are *not* waste, allowing use of segment.Set gap-tracking to efficiently
	// find ranges for both reclaim and recycling allocations. Consequently, the
	// value type of unwasteSet is an empty struct.
	//
	// unwasteSmall and unwasteHuge are protected by mu.
	unwasteSmall unwasteSet
	unwasteHuge  unwasteSet

	// reclaimable is true if there may be at least one waste page in the
	// MemoryFile.
	//
	// reclaimable is protected by mu.
	reclaimable bool

	// reclaimCond is signaled (with mu locked) when reclaimable or destroyed
	// transitions from false to true.
	reclaimCond sync.Cond

	// unfreeSmall and unfreeHuge track information for void, used, waste,
	// reclaiming, and sub-reclaimed ranges backed by small/huge pages
	// respectively. Each unfreeSet also contains segments representing chunks
	// that are backed by a different page size. Gaps in the sets therefore
	// represent free ranges backed by small/huge pages.
	//
	// unfreeSmall and unfreeHuge are protected by mu.
	unfreeSmall unfreeSet
	unfreeHuge  unfreeSet

	// subreclaimed maps hugepage-aligned file offsets to the number of
	// sub-reclaimed small pages within the hugepage beginning at that offset.
	// subreclaimed is protected by mu.
	subreclaimed map[uint64]uint64

	// These fields are used for memory accounting.
	//
	// Memory accounting is based on identifying the set of committed pages.
	// Since we do not have direct access to the MMU, tracking application
	// accesses to uncommitted pages to detect commitment would introduce
	// additional page faults, which would be prohibitively expensive. Instead,
	// we query the host kernel to determine which pages are committed.
	//
	// knownCommitment tracks ranges that are known to be committed or not
	// committed. Gaps in knownCommitment indicate ranges whose commitment is
	// unknown.
	//
	// knownCommittedBytes is the number of bytes in the file known to be
	// committed, i.e. the span of all segments in knownCommitment for which
	// committed == true.
	//
	// commitSeq is a sequence counter used to detect races between scans for
	// committed pages and reallocation.
	//
	// nextCommitScan is the next time at which UpdateUsage() may scan the
	// backing file for commitment information.
	//
	// All of these fields are protected by mu.
	knownCommitment     commitmentSet
	knownCommittedBytes uint64
	commitSeq           uint64
	nextCommitScan      time.Time

	// evictable maps EvictableMemoryUsers to eviction state.
	//
	// evictable is protected by mu.
	evictable map[EvictableMemoryUser]*evictableMemoryUserInfo

	// evictionWG counts the number of goroutines currently performing evictions.
	evictionWG sync.WaitGroup

	// opts holds options passed to NewMemoryFile. opts is immutable.
	opts MemoryFileOpts

	// destroyed is set by Destroy to instruct the reclaimer goroutine to
	// release resources and exit. destroyed is protected by mu.
	destroyed bool

	// stopNotifyPressure stops memory cgroup pressure level
	// notifications used to drive eviction. stopNotifyPressure is
	// immutable.
	stopNotifyPressure func()

	// file is the backing file. The file pointer is immutable.
	file *os.File

	// chunks holds metadata for each usable chunk in the backing file.
	//
	// chunks is at the end of MemoryFile in hopes of placing it on a relatively
	// quiet cache line, since MapInternal() is by far the hottest path through
	// pgalloc.
	//
	// Reading chunks requires either that chunksSeq is in a reader critical
	// section or that mu is locked. Mutating chunks requires both that
	// chunksSeq is in a writer critical section and that mu is locked.
	chunksSeq sync.SeqCount `state:"nosave"`
	chunks    []chunkInfo
}

const (
	chunkShift = 30
	chunkSize  = 1 << chunkShift // 1 GB
	chunkMask  = chunkSize - 1
	maxChunks  = math.MaxInt64 / chunkSize // because file size is int64
	// Note that the maxChunks limit means that hostarch.[Huge]PageRoundUp() on
	// allocated file offsets can't overflow, so we can use the Must* variants.
)

// chunkInfo is the value type of MemoryFile.chunks.
//
// +stateify savable
type chunkInfo struct {
	// huge is true if this chunk is expected to be hugepage-backed and false if
	// this chunk is expected to be smallpage-backed.
	//
	// huge is immutable.
	huge bool

	// mapping is the start address of a mapping of the chunk.
	//
	// mapping is immutable.
	mapping uintptr `state:"nosave"`
}

func (f *MemoryFile) chunksLoad() []chunkInfo {
	return SeqAtomicLoadChunkInfoSlice(&f.chunksSeq, &f.chunks)
}

// forEachChunk invokes fn on a sequence of chunks that collectively span all
// bytes in fr. In each call, chunkFR is the subset of fr that falls within
// chunk.
func (f *MemoryFile) forEachChunk(fr memmap.FileRange, fn func(chunk *chunkInfo, chunkFR memmap.FileRange)) {
	chunks := f.chunksLoad()
	chunkStart := fr.Start &^ chunkMask
	i := int(fr.Start / chunkSize)
	for chunkStart < fr.End {
		chunkEnd := chunkStart + chunkSize
		fn(&chunks[i], fr.Intersect(memmap.FileRange{chunkStart, chunkEnd}))
		chunkStart = chunkEnd
		i++
	}
}

// Preconditions: The FileRange represented by c contains off.
func (c *chunkInfo) mappingAt(off uint64) uintptr {
	return c.mapping + uintptr(off&chunkMask)
}

// unwasteInfo is the value type of MemoryFile.unwasteSmall/Huge.
//
// +stateify savable
type unwasteInfo struct{}

// unfreeInfo is the value type of MemoryFile.unfreeSmall/Huge.
//
// +stateify savable
type unfreeInfo struct {
	// refs is the per-page reference count. refs is non-zero for used pages, and
	// zero for void, waste, reclaiming, and sub-reclaimed pages, as well as
	// pages backed by a different page size.
	refs uint64

	// kind is the memory accounting type. kind is allocation-dependent for used
	// pages, and usage.System for void, waste, reclaiming, and sub-reclaimed
	// pages, as well as pages backed by a different page size.
	kind usage.MemoryKind

	// If this segment represents used pages, commitSeq was the value of
	// MemoryFile.commitSeq when the pages last transitioned from free to used,
	// or when the pages were last decommitted using Decommit().
	commitSeq uint64
}

// commitmentInfo is the value type of MemoryFile.knownCommitment.
//
// +stateify savable
type commitmentInfo struct {
	// If committed is true, the represented range is known to be committed.
	// If committed is false, the represented range is known to be uncommitted.
	committed bool
}

// An EvictableMemoryUser represents a user of MemoryFile-allocated memory that
// may be asked to deallocate that memory in the presence of memory pressure.
type EvictableMemoryUser interface {
	// Evict requests that the EvictableMemoryUser deallocate memory used by
	// er, which was registered as evictable by a previous call to
	// MemoryFile.MarkEvictable.
	//
	// Evict is not required to deallocate memory. In particular, since pgalloc
	// must call Evict without holding locks to avoid circular lock ordering,
	// it is possible that the passed range has already been marked as
	// unevictable by a racing call to MemoryFile.MarkUnevictable.
	// Implementations of EvictableMemoryUser must detect such races and handle
	// them by making Evict have no effect on unevictable ranges.
	//
	// After a call to Evict, the MemoryFile will consider the evicted range
	// unevictable (i.e. it will not call Evict on the same range again) until
	// informed otherwise by a subsequent call to MarkEvictable.
	Evict(ctx context.Context, er EvictableRange)
}

// An EvictableRange represents a range of uint64 offsets in an
// EvictableMemoryUser.
//
// In practice, most EvictableMemoryUsers will probably be implementations of
// memmap.Mappable, and EvictableRange therefore corresponds to
// memmap.MappableRange. However, this package cannot depend on the memmap
// package, since doing so would create a circular dependency.
//
// type EvictableRange <generated using go_generics>

// evictableMemoryUserInfo is the value type of MemoryFile.evictable.
type evictableMemoryUserInfo struct {
	// ranges tracks all evictable ranges for the given user.
	ranges evictableRangeSet

	// If evicting is true, there is a goroutine currently evicting all
	// evictable ranges for this user.
	evicting bool
}

// MemoryFileOpts provides options to NewMemoryFile.
type MemoryFileOpts struct {
	// DelayedEviction controls the extent to which the MemoryFile may delay
	// eviction of evictable allocations.
	DelayedEviction DelayedEvictionType

	// If UseHostMemcgPressure is true, use host memory cgroup pressure level
	// notifications to determine when eviction is necessary. This option has
	// no effect unless DelayedEviction is DelayedEvictionEnabled.
	UseHostMemcgPressure bool

	// If DisableIMAWorkAround is true, NewMemoryFile will not call
	// IMAWorkAroundForMemFile().
	DisableIMAWorkAround bool

	// DecommitOnDestroy indicates whether the entire host file should be
	// decommitted on destruction. This is appropriate for host filesystem based
	// files that need to be explicitly cleaned up to release disk space.
	DecommitOnDestroy bool

	// If ExpectHugepages is true, MemoryFile will expect that the host will
	// attempt to back hugepage-aligned ranges, with huge pages explicitly
	// requested if AdviseHugepage is true, with huge pages. If ExpectHugepages
	// is false, MemoryFile will expect that the host will back all allocations
	// with small pages.
	ExpectHugepages bool

	// If AdviseHugepage is true, MemoryFile will explicitly request that the
	// host back AllocOpts.Hugepage == true allocations with huge pages.
	AdviseHugepage bool

	// If AdviseNoHugepage is true, MemoryFile will explicitly request that the
	// host back AllocOpts.Hugepage == false allocations with small pages.
	AdviseNoHugepage bool
}

// DelayedEvictionType is the type of MemoryFileOpts.DelayedEviction.
type DelayedEvictionType uint8

const (
	// DelayedEvictionDefault has unspecified behavior.
	DelayedEvictionDefault DelayedEvictionType = iota

	// DelayedEvictionDisabled requires that evictable allocations are evicted
	// as soon as possible.
	DelayedEvictionDisabled

	// DelayedEvictionEnabled requests that the MemoryFile delay eviction of
	// evictable allocations until doing so is considered necessary to avoid
	// performance degradation due to host memory pressure, or OOM kills.
	//
	// As of this writing, the behavior of DelayedEvictionEnabled depends on
	// whether or not MemoryFileOpts.UseHostMemcgPressure is enabled:
	//
	//	- If UseHostMemcgPressure is true, evictions are delayed until memory
	//	pressure is indicated.
	//
	//	- Otherwise, evictions are only delayed until the reclaimer goroutine
	//	is out of work (pages to reclaim).
	DelayedEvictionEnabled

	// DelayedEvictionManual requires that evictable allocations are only
	// evicted when MemoryFile.StartEvictions() is called. This is extremely
	// dangerous outside of tests.
	DelayedEvictionManual
)

// NewMemoryFile creates a MemoryFile backed by the given file. If
// NewMemoryFile succeeds, ownership of file is transferred to the returned
// MemoryFile.
func NewMemoryFile(file *os.File, opts MemoryFileOpts) (*MemoryFile, error) {
	switch opts.DelayedEviction {
	case DelayedEvictionDefault:
		opts.DelayedEviction = DelayedEvictionEnabled
	case DelayedEvictionDisabled, DelayedEvictionManual:
		opts.UseHostMemcgPressure = false
	case DelayedEvictionEnabled:
		// ok
	default:
		return nil, fmt.Errorf("invalid MemoryFileOpts.DelayedEviction: %v", opts.DelayedEviction)
	}

	// Truncate the file to 0 bytes first to ensure that it's empty.
	if err := file.Truncate(0); err != nil {
		return nil, err
	}
	f := &MemoryFile{
		subreclaimed: make(map[uint64]uint64),
		evictable:    make(map[EvictableMemoryUser]*evictableMemoryUserInfo),
		opts:         opts,
		file:         file,
	}
	f.reclaimCond.L = &f.mu
	// Initially, all pages are void.
	fullFR := memmap.FileRange{0, math.MaxUint64}
	f.unwasteSmall.InsertRange(fullFR, unwasteInfo{})
	f.unwasteHuge.InsertRange(fullFR, unwasteInfo{})
	f.unfreeSmall.InsertRange(fullFR, unfreeInfo{
		kind: usage.System,
	})
	f.unfreeHuge.InsertRange(fullFR, unfreeInfo{
		kind: usage.System,
	})
	f.knownCommitment.InsertRange(fullFR, commitmentInfo{
		committed: false,
	})

	if f.opts.DelayedEviction == DelayedEvictionEnabled && f.opts.UseHostMemcgPressure {
		stop, err := hostmm.NotifyCurrentMemcgPressureCallback(func() {
			f.mu.Lock()
			startedAny := f.startEvictionsLocked()
			f.mu.Unlock()
			if startedAny {
				log.Debugf("pgalloc.MemoryFile performing evictions due to memcg pressure")
			}
		}, "low")
		if err != nil {
			return nil, fmt.Errorf("failed to configure memcg pressure level notifications: %v", err)
		}
		f.stopNotifyPressure = stop
	}

	go f.runReclaim() // S/R-SAFE: f.mu

	if !opts.DisableIMAWorkAround {
		IMAWorkAroundForMemFile(file.Fd())
	}
	return f, nil
}

// IMAWorkAroundForMemFile works around IMA by immediately creating a temporary
// PROT_EXEC mapping, while the backing file is still small. IMA will ignore
// any future mappings.
//
// The Linux kernel contains an optional feature called "Integrity
// Measurement Architecture" (IMA). If IMA is enabled, it will checksum
// binaries the first time they are mapped PROT_EXEC. This is bad news for
// executable pages mapped from our backing file, which can grow to
// terabytes in (sparse) size. If IMA attempts to checksum a file that
// large, it will allocate all of the sparse pages and quickly exhaust all
// memory.
func IMAWorkAroundForMemFile(fd uintptr) {
	m, _, errno := unix.Syscall6(
		unix.SYS_MMAP,
		0,
		hostarch.PageSize,
		unix.PROT_EXEC,
		unix.MAP_SHARED,
		fd,
		0)
	if errno != 0 {
		// This isn't fatal (IMA may not even be in use). Log the error, but
		// don't return it.
		log.Warningf("Failed to pre-map MemoryFile PROT_EXEC: %v", errno)
	} else {
		if _, _, errno := unix.Syscall(
			unix.SYS_MUNMAP,
			m,
			hostarch.PageSize,
			0); errno != 0 {
			panic(fmt.Sprintf("failed to unmap PROT_EXEC MemoryFile mapping: %v", errno))
		}
	}
}

// Destroy releases all resources used by f.
//
// Preconditions: All pages allocated by f have been freed.
//
// Postconditions: None of f's methods may be called after Destroy.
func (f *MemoryFile) Destroy() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.destroyed = true
	f.reclaimCond.Signal()
}

// AllocOpts are options used in MemoryFile.Allocate.
type AllocOpts struct {
	// Kind is the allocation's memory accounting type.
	Kind usage.MemoryKind

	// If Hugepage is true, the allocation should be hugepage-backed if possible.
	Hugepage bool

	// Hint indicates the caller's expectations regarding the near-future use of
	// the allocated range.
	Hint HintType

	// If TryPopulateInternal is true, the caller expects to access the
	// allocation's contents through MapInternal() soon, so MemoryFile should
	// opportunistically map backing pages into the caller's page tables. Failure
	// to do so does not cause allocation to fail.
	TryPopulateInternal bool

	// Dir indicates the direction in which consecutive allocations (without
	// intervening deallocations) return offsets.
	Dir Direction
}

// Direction describes how to allocate offsets from MemoryFile.
type Direction uint8

const (
	// BottomUp allocates offsets in increasing offsets.
	BottomUp Direction = iota
	// TopDown allocates offsets in decreasing offsets.
	TopDown
)

// String implements fmt.Stringer.
func (d Direction) String() string {
	switch d {
	case BottomUp:
		return "up"
	case TopDown:
		return "down"
	}
	panic(fmt.Sprintf("invalid direction: %d", d))
}

// HintType is the type of AllocOpts.Hint.
type HintType uint8

const (
	// HintNone is a no-op.
	HintNone HintType = iota

	// HintOne indicates that at least one page in the allocation is expected to
	// be used immediately.
	HintOne

	// HintAll indicates that all pages in the allocation are expected to be used
	// immediately.
	HintAll
)

type allocateState struct {
	length      uint64
	opts        AllocOpts
	huge        bool
	willCommit  bool
	needZeroing bool
}

// Allocate returns a range of initially-zeroed pages of the given length, with
// a single reference on each page held by the caller. When the last reference
// on an allocated page is released, ownership of the page is returned to the
// MemoryFile, allowing it to be returned by a future call to Allocate.
//
// Preconditions:
//   - length > 0.
//   - length must be page-aligned.
//   - If opts.Hugepage == true, length must be hugepage-aligned.
func (f *MemoryFile) Allocate(length uint64, opts AllocOpts) (memmap.FileRange, error) {
	if length == 0 || !hostarch.IsPageAligned(length) || (opts.Hugepage && !hostarch.IsHugePageAligned(length)) {
		panic(fmt.Sprintf("invalid allocation length: %#x", length))
	}

	alloc := allocateState{
		length: length,
		opts:   opts,
		huge:   opts.Hugepage && f.opts.ExpectHugepages,
	}

	// Determine if the caller expects to immediately use all pages in this
	// allocation. If so, we can recycle waste pages when allocating without
	// increasing memory usage.
	if opts.Hint == HintAll {
		alloc.willCommit = true
	} else if opts.Hint == HintOne && (length == hostarch.PageSize || (alloc.huge && length == hostarch.HugePageSize)) {
		alloc.willCommit = true
	}

	// Allocate pages.
	fr, err := f.findAllocatableAndMarkUsed(&alloc)
	if err != nil {
		return fr, err
	}

	if alloc.willCommit {
		// Reminder: The caller expects all pages in this allocation to be used
		// immediately.
		if alloc.needZeroing {
			// We will need page table entries in our address space to zero these
			// pages.
			opts.TryPopulateInternal = true
		}
		needHugeTouch := false
		if !opts.TryPopulateInternal && alloc.huge && f.opts.AdviseHugepage {
			// If we do nothing, the first access to the allocation will be through a
			// platform.AddressSpace, which does not have MADV_HUGEPAGE (=> vma flag
			// VM_HUGEPAGE) set. Consequently, shmem_fault() => shmem_getpage_gfp() /
			// shmem_get_folio_gfp() will commit a small page. khugepaged may
			// eventually collapse the containing hugepage-aligned region into a huge
			// page when it scans our mapping (khugepaged_scan_mm_slot() =>
			// khugepaged_scan_file()), but this depends on khugepaged_max_ptes_none,
			// and in addition to the latency and overhead of doing so, this will
			// incur another round of page faults.
			//
			// If mlock()ing through our mappings succeeds, then it will avoid this
			// problem. Otherwise, touch each hugepage through our mappings.
			opts.TryPopulateInternal = true
			needHugeTouch = true
		}
		if opts.TryPopulateInternal {
			// mlock(2) populates pages (commits them and maps them into our page
			// tables); munlock(2) cancels the mlock but leaves pages populated. This
			// is fairly expensive (two host syscalls plus the host MM overhead of
			// splitting and merging VMAs for mlock + munlock), so only do so for
			// allocations greater than 2 hugepages (>= 3 host page faults) in size.
			populated := false
			if canMlock() && length > 2*hostarch.HugePageSize {
				f.forEachChunk(fr, func(chunk *chunkInfo, chunkFR memmap.FileRange) {
					if !canMlock() {
						return
					}
					addr := chunk.mappingAt(chunkFR.Start)
					maplen := uintptr(chunkFR.Length())
					_, _, errno := unix.Syscall(unix.SYS_MLOCK, addr, maplen, 0)
					unix.Syscall(unix.SYS_MUNLOCK, addr, maplen, 0)
					if errno != 0 {
						if errno == unix.ENOMEM || errno == unix.EPERM {
							// These errors are expected from hitting non-zero RLIMIT_MEMLOCK,
							// or hitting zero RLIMIT_MEMLOCK without CAP_IPC_LOCK,
							// respectively.
							log.Infof("Disabling mlock: %s", errno)
						} else {
							log.Warningf("Disabling mlock: %s", errno)
						}
						mlockDisabled.Store(1)
					}
					// If mlocking failed at any point, canMlock() is now false.
					populated = canMlock()
				})
			}
			if needHugeTouch && !populated {
				f.forEachMappingSlice(fr, func(bs []byte) {
					for i := 0; i < len(bs); i += hostarch.HugePageSize {
						bs[i] = 0
					}
				})
			}
		}
	}

	// Zero out pages if their contents are unknown.
	if alloc.needZeroing {
		f.manuallyZero(fr)
	}

	return fr, nil
}

var mlockDisabled atomicbitops.Uint32

func canMlock() bool {
	return mlockDisabled.Load() == 0
}

func (f *MemoryFile) manuallyZero(fr memmap.FileRange) {
	f.forEachMappingSlice(fr, func(bs []byte) {
		for i := range bs {
			bs[i] = 0
		}
	})
}

func (f *MemoryFile) findAllocatableAndMarkUsed(alloc *allocateState) (fr memmap.FileRange, err error) {
	unwaste := &f.unwasteSmall
	unfree := &f.unfreeSmall
	initRefs := uint64(1)
	if alloc.huge {
		unwaste = &f.unwasteHuge
		unfree = &f.unfreeHuge
		initRefs = uint64(pagesPerHugePage)
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	if alloc.willCommit {
		// Try to recycle waste pages, since this avoids the overhead of
		// decommitting and then committing them again.
		var uwgap unwasteGapIterator
		if alloc.opts.Dir == BottomUp {
			uwgap = unwaste.FirstLargeEnoughGap(alloc.length)
		} else {
			uwgap = unwaste.LastLargeEnoughGap(alloc.length)
		}
		if uwgap.Ok() {
			if alloc.opts.Dir == BottomUp {
				fr = memmap.FileRange{
					Start: uwgap.Start(),
					End:   uwgap.Start() + alloc.length,
				}
			} else {
				fr = memmap.FileRange{
					Start: uwgap.End() - alloc.length,
					End:   uwgap.End(),
				}
			}
			alloc.needZeroing = true
			unwaste.Insert(uwgap, fr, unwasteInfo{})
			// Replace segments in unfree spanning fr (there may be more than one due
			// to differing commitSeqs) with one for our allocation.
			ufgap := unfree.RemoveRange(fr)
			unfree.Insert(ufgap, fr, unfreeInfo{
				refs:      1,
				kind:      alloc.opts.Kind,
				commitSeq: f.commitSeq,
			})
			// These pages may be known-committed accounted to usage.System (from
			// when they became waste), unknown-commitment, or known-decommitted (if
			// UpdateUsage() scanned them after becoming waste); convert them all to
			// known-committed accounted to our allocation kind, reflecting
			// precommit.
			kcseg, kcgap := f.knownCommitment.Find(fr.Start)
		AccountingUpdateLoop:
			for {
				switch {
				case kcseg.Ok() && kcseg.Start() < fr.End:
					kcfr := kcseg.Range().Intersect(fr)
					if kcseg.ValuePtr().committed {
						usage.MemoryAccounting.Move(kcfr.Length(), alloc.opts.Kind, usage.System)
					} else {
						kcseg = f.knownCommitment.Isolate(kcseg, kcfr)
						kcseg.ValuePtr().committed = true
						f.knownCommittedBytes += kcfr.Length()
						usage.MemoryAccounting.Inc(kcfr.Length(), alloc.opts.Kind)
						kcseg = f.knownCommitment.MergePrev(kcseg)
					}
					kcseg, kcgap = kcseg.NextNonEmpty()
				case kcgap.Ok() && kcgap.Start() < fr.End:
					kcfr := kcgap.Range().Intersect(fr)
					kcseg = f.knownCommitment.Insert(kcgap, kcfr, commitmentInfo{committed: true})
					f.knownCommittedBytes += kcfr.Length()
					usage.MemoryAccounting.Inc(kcfr.Length(), alloc.opts.Kind)
					kcseg, kcgap = kcseg.NextNonEmpty()
				default:
					f.knownCommitment.MergePrev(kcseg)
					break AccountingUpdateLoop
				}
			}
			return
		}
	}
	// No suitable waste pages or we can't use them.

	for {
		// Try to allocate free pages from existing chunks.
		var ufgap unfreeGapIterator
		if alloc.opts.Dir == BottomUp {
			ufgap = unfree.FirstLargeEnoughGap(alloc.length)
		} else {
			ufgap = unfree.LastLargeEnoughGap(alloc.length)
		}
		if ufgap.Ok() {
			if alloc.opts.Dir == BottomUp {
				fr = memmap.FileRange{
					Start: ufgap.Start(),
					End:   ufgap.Start() + alloc.length,
				}
			} else {
				fr = memmap.FileRange{
					Start: ufgap.End() - alloc.length,
					End:   ufgap.End(),
				}
			}
			unfree.InsertRange(fr, unfreeInfo{
				refs:      initRefs,
				kind:      alloc.opts.Kind,
				commitSeq: f.commitSeq,
			})
			// These pages should all be known-decommitted.
			kcseg := f.knownCommitment.FindSegment(fr.Start)
			kcseg = f.knownCommitment.Isolate(kcseg, fr)
			if alloc.willCommit {
				// Mark them known-committed to reflect that we are about to commit
				// them.
				kcseg.ValuePtr().committed = true
				f.knownCommittedBytes += fr.Length()
				usage.MemoryAccounting.Inc(fr.Length(), alloc.opts.Kind)
				f.knownCommitment.MergeAdjacent(kcseg)
			} else {
				// Mark them unknown-commitment, since the allocated memory can be
				// committed by the allocation's users at any time without MemoryFile's
				// knowledge.
				f.knownCommitment.Remove(kcseg)
			}
			return
		}

		// Extend the file to create more chunks.
		err = f.extendChunksLocked(alloc)
		if err != nil {
			return
		}

		// Retry the allocation using new chunks.
	}
}

// Preconditions: f.mu must be locked.
func (f *MemoryFile) extendChunksLocked(alloc *allocateState) error {
	unfree := &f.unfreeSmall
	if alloc.huge {
		unfree = &f.unfreeHuge
	}

	oldChunks := f.chunksLoad()
	oldNrChunks := uint64(len(oldChunks))
	oldFileSize := oldNrChunks * chunkSize

	// Determine how many chunks we need to satisfy alloc.
	tail := uint64(0)
	if oldNrChunks != 0 {
		if lastChunk := oldChunks[oldNrChunks-1]; lastChunk.huge == alloc.huge {
			// We can use free pages at the end of the current last chunk.
			if ufgap := unfree.FindGap(oldFileSize - 1); ufgap.Ok() {
				tail = ufgap.Range().Length()
			}
		}
	}
	incNrChunks := (alloc.length + chunkMask - tail) / chunkSize
	incFileSize := incNrChunks * chunkSize
	newNrChunks := oldNrChunks + incNrChunks
	if newNrChunks > maxChunks || newNrChunks < oldNrChunks /* overflow */ {
		return linuxerr.ENOMEM
	}
	newFileSize := newNrChunks * chunkSize

	// Extend the backing file.
	if err := f.file.Truncate(int64(newFileSize)); err != nil {
		return err
	}

	// Obtain mappings for the new chunks.
	var mapStart uintptr
	if alloc.huge {
		// Ensure that this mapping is hugepage-aligned.
		m, err := memutil.MapAlignedPrivateAnon(uintptr(incFileSize), hostarch.HugePageSize, unix.PROT_NONE, 0)
		if err != nil {
			return err
		}
		_, _, errno := unix.Syscall6(
			unix.SYS_MMAP,
			m,
			uintptr(incFileSize),
			unix.PROT_READ|unix.PROT_WRITE,
			unix.MAP_SHARED|unix.MAP_FIXED,
			f.file.Fd(),
			uintptr(oldFileSize))
		if errno != 0 {
			unix.RawSyscall(unix.SYS_MUNMAP, m, uintptr(incFileSize), 0)
			return errno
		}
		mapStart = m
	} else {
		m, _, errno := unix.Syscall6(
			unix.SYS_MMAP,
			0,
			uintptr(incFileSize),
			unix.PROT_READ|unix.PROT_WRITE,
			unix.MAP_SHARED,
			f.file.Fd(),
			uintptr(oldFileSize))
		if errno != 0 {
			return errno
		}
		mapStart = m
	}
	f.adviseChunkMapping(mapStart, uintptr(incFileSize), alloc.huge)

	// Update chunk state.
	newChunks := make([]chunkInfo, newNrChunks, newNrChunks)
	copy(newChunks, oldChunks)
	m := mapStart
	for i := oldNrChunks; i < newNrChunks; i++ {
		newChunks[i].huge = alloc.huge
		newChunks[i].mapping = m
		m += chunkSize
	}
	f.chunksSeq.BeginWrite()
	f.chunks = newChunks
	f.chunksSeq.EndWrite()

	// Mark void pages free.
	unfree.RemoveRange(memmap.FileRange{
		Start: oldNrChunks * chunkSize,
		End:   newNrChunks * chunkSize,
	})

	return nil
}

func (f *MemoryFile) adviseChunkMapping(addr, len uintptr, huge bool) {
	if huge {
		if f.opts.AdviseHugepage {
			_, _, errno := unix.Syscall(unix.SYS_MADVISE, addr, len, unix.MADV_HUGEPAGE)
			if errno != 0 {
				// Log this failure but continue.
				log.Warningf("madvise(%#x, %d, MADV_HUGEPAGE) failed: %s", addr, len, errno)
			}
		}
	} else {
		if f.opts.AdviseNoHugepage {
			_, _, errno := unix.Syscall(unix.SYS_MADVISE, addr, len, unix.MADV_NOHUGEPAGE)
			if errno != 0 {
				// Log this failure but continue.
				log.Warningf("madvise(%#x, %d, MADV_NOHUGEPAGE) failed: %s", addr, len, errno)
			}
		}
	}
}

// AllocateAndFill allocates memory of the given kind and fills it by calling
// r.ReadToBlocks() repeatedly until either length bytes are read or a non-nil
// error is returned. It returns the memory filled by r, truncated down to the
// nearest page. If this is shorter than length bytes due to an error returned
// by r.ReadToBlocks(), it returns that error.
//
// Most callers will want to pass opts.TryPopulateInternal == true.
//
// Preconditions: As for Allocate.
func (f *MemoryFile) AllocateAndFill(length uint64, opts AllocOpts, r safemem.Reader) (memmap.FileRange, error) {
	fr, err := f.Allocate(length, opts)
	if err != nil {
		return memmap.FileRange{}, err
	}
	dsts, err := f.MapInternal(fr, hostarch.Write)
	if err != nil {
		f.DecRef(fr)
		return memmap.FileRange{}, err
	}
	n, err := safemem.ReadFullToBlocks(r, dsts)
	un := hostarch.PageRoundDown(n)
	if un < length {
		// Free unused memory and update fr to contain only the memory that is
		// still allocated.
		f.DecRef(memmap.FileRange{fr.Start + un, fr.End})
		fr.End = fr.Start + un
	}
	return fr, err
}

// IncRef implements memmap.File.IncRef.
func (f *MemoryFile) IncRef(fr memmap.FileRange) {
	if !fr.WellFormed() || fr.Length() == 0 || !hostarch.IsPageAligned(fr.Start) || !hostarch.IsPageAligned(fr.End) {
		panic(fmt.Sprintf("invalid range: %v", fr))
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	f.forEachChunk(fr, func(chunk *chunkInfo, chunkFR memmap.FileRange) {
		unfree := &f.unfreeSmall
		if chunk.huge {
			unfree = &f.unfreeHuge
		}
		ufseg := unfree.LowerBoundSegmentSplitBefore(chunkFR.Start)
		for ufseg.Ok() && ufseg.Start() < chunkFR.End {
			ufseg = unfree.SplitAfter(ufseg, chunkFR.End)
			ufseg.ValuePtr().refs++
			ufseg = unfree.MergePrev(ufseg).NextSegment()
		}
		unfree.MergePrev(ufseg)
	})
}

// DecRef implements memmap.File.DecRef.
func (f *MemoryFile) DecRef(fr memmap.FileRange) {
	if !fr.WellFormed() || fr.Length() == 0 || !hostarch.IsPageAligned(fr.Start) || !hostarch.IsPageAligned(fr.End) {
		panic(fmt.Sprintf("invalid range: %v", fr))
	}

	f.mu.Lock()
	defer f.mu.Unlock()

	reclaimable := false

	f.forEachChunk(fr, func(chunk *chunkInfo, chunkFR memmap.FileRange) {
		unwaste := &f.unwasteSmall
		unfree := &f.unfreeSmall
		if chunk.huge {
			unwaste = &f.unwasteHuge
			unfree = &f.unfreeHuge
		}
		ufseg := unfree.LowerBoundSegmentSplitBefore(chunkFR.Start)
		for ufseg.Ok() && ufseg.Start() < chunkFR.End {
			ufseg = unfree.SplitAfter(ufseg, chunkFR.End)
			uf := ufseg.ValuePtr()
			uf.refs--
			if uf.refs == 0 {
				// Mark these pages as waste.
				wasteFR := ufseg.Range()
				unwaste.RemoveRange(wasteFR)
				reclaimable = true
				// Reclassify waste memory as System until it's recycled or reclaimed.
				committed := uint64(0)
				for kcseg := f.knownCommitment.LowerBoundSegment(wasteFR.Start); kcseg.Ok() && kcseg.Start() < wasteFR.End; kcseg = kcseg.NextSegment() {
					if kcseg.ValuePtr().committed {
						committed += kcseg.Range().Intersect(wasteFR).Length()
					}
				}
				usage.MemoryAccounting.Move(committed, usage.System, uf.kind)
				uf.kind = usage.System
			}
			ufseg = unfree.MergePrev(ufseg).NextSegment()
		}
		unfree.MergePrev(ufseg)
	})

	// Wake the reclaimer if we marked any pages as waste. Leave this for just
	// before we unlock f.mu.
	if reclaimable && !f.reclaimable {
		f.reclaimable = true
		f.reclaimCond.Signal()
	}
}

// runReclaim implements the reclaimer goroutine.
func (f *MemoryFile) runReclaim() {
	for {
		fr, ok := f.findWasteAndMarkReclaiming()
		if !ok {
			break
		}
		f.decommitFile(fr)
		f.markReclaimed(fr)
	}

	// We only get here if findWasteAndMarkReclaiming finds f.destroyed set and
	// returns false.
	f.mu.Lock()
	if !f.destroyed {
		f.mu.Unlock()
		panic("findReclaimable broke out of reclaim loop, but destroyed is no longer set")
	}
	f.file.Close()
	// Ensure that any attempts to use f.file.Fd() fail instead of getting a fd
	// that has possibly been reassigned.
	f.file = nil
	chunks := f.chunksLoad()
	for i := range chunks {
		chunk := &chunks[i]
		_, _, errno := unix.Syscall(unix.SYS_MUNMAP, chunk.mapping, chunkSize, 0)
		if errno != 0 {
			log.Warningf("Failed to unmap mapping %#x for MemoryFile chunk %d: %v", chunk.mapping, i, errno)
		}
		chunk.mapping = 0
	}
	f.mu.Unlock()

	// This must be called without holding f.mu to avoid circular lock
	// ordering.
	if f.stopNotifyPressure != nil {
		f.stopNotifyPressure()
	}
}

func (f *MemoryFile) findWasteAndMarkReclaiming() (memmap.FileRange, bool) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for {
		for {
			if f.destroyed {
				return memmap.FileRange{}, false
			}
			if f.reclaimable {
				break
			}
			if f.opts.DelayedEviction == DelayedEvictionEnabled && !f.opts.UseHostMemcgPressure {
				// No work to do. Evict any pending evictable allocations to
				// get more reclaimable pages before going to sleep.
				f.startEvictionsLocked()
			}
			f.reclaimCond.Wait() // temporarily releases f.mu
		}
		// Hugepages are relatively rare and expensive due to fragmentation and the
		// cost of compaction. Most allocations are done upwards, with exceptions
		// being stacks and some allocators that allocate top-down. So we expect
		// lower offsets to weakly correlate with older allocations, which are more
		// likely to actually be hugepage-backed. Thus, reclaim from unwasteSmall
		// before unwasteHuge, and higher offsets before lower ones.
		for _, unwaste := range [...]*unwasteSet{&f.unwasteSmall, &f.unwasteHuge} {
			if uwgap := unwaste.LastLargeEnoughGap(1); uwgap.Ok() {
				fr := uwgap.Range()
				// Linux serializes fallocate()s on shmem files, so limit the amount we
				// reclaim at once to avoid starving Decommit().
				const maxReclaimingBytes = 128 << 20 // 128 MB
				if fr.Length() > maxReclaimingBytes {
					fr.Start = fr.End - maxReclaimingBytes
				}
				// Mark this range as reclaiming (no longer waste, ineligible for
				// recycling).
				unwaste.Insert(uwgap, fr, unwasteInfo{})
				// Mark this range known-uncommitted immediately (rather than after
				// decommitting completes) so that UpdateUsage() will skip scanning it
				// while it's being decommitted.
				f.markReclaimingKnownUncommittedLocked(fr)
				return fr, true
			}
		}
		// Nothing is reclaimable.
		f.reclaimable = false
	}
}

// Preconditions: f.mu must be locked.
func (f *MemoryFile) markReclaimingKnownUncommittedLocked(fr memmap.FileRange) {
	kcseg, kcgap := f.knownCommitment.Find(fr.Start)
	for {
		switch {
		case kcseg.Ok() && kcseg.Start() < fr.End:
			kcfr := kcseg.Range().Intersect(fr)
			if kcseg.ValuePtr().committed {
				kcseg = f.knownCommitment.Isolate(kcseg, kcfr)
				kcseg.ValuePtr().committed = false
				f.knownCommittedBytes -= kcfr.Length()
				// Waste memory always has accounting type usage.System.
				usage.MemoryAccounting.Dec(kcfr.Length(), usage.System)
				kcseg = f.knownCommitment.MergePrev(kcseg)
			}
			kcseg, kcgap = kcseg.NextNonEmpty()
		case kcgap.Ok() && kcgap.Start() < fr.End:
			kcfr := kcgap.Range().Intersect(fr)
			kcseg = f.knownCommitment.Insert(kcgap, kcfr, commitmentInfo{committed: false})
			kcseg, kcgap = f.knownCommitment.MergePrev(kcseg).NextNonEmpty()
		default:
			if kcseg.Ok() {
				f.knownCommitment.MergePrev(kcseg)
			}
			return
		}
	}
}

func (f *MemoryFile) decommitFile(fr memmap.FileRange) {
	// "After a successful call, subsequent reads from this range will
	// return zeroes. The FALLOC_FL_PUNCH_HOLE flag must be ORed with
	// FALLOC_FL_KEEP_SIZE in mode ..." - fallocate(2)
	err := unix.Fallocate(
		int(f.file.Fd()),
		unix.FALLOC_FL_PUNCH_HOLE|unix.FALLOC_FL_KEEP_SIZE,
		int64(fr.Start),
		int64(fr.Length()))
	if err != nil {
		log.Warningf("Failed to decommit %v: %v", fr, err)
		// Zero the pages manually. This won't reduce memory usage, but at least
		// ensures that the pages have the right contents.
		f.manuallyZero(fr)
	}
}

func (f *MemoryFile) markReclaimed(fr memmap.FileRange) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.forEachChunk(fr, func(chunk *chunkInfo, chunkFR memmap.FileRange) {
		if !chunk.huge {
			f.unfreeSmall.RemoveRange(chunkFR)
			return
		}

		firstHugeStart := hostarch.HugePageRoundDown(chunkFR.Start)
		if firstHugeStart == hostarch.HugePageRoundDown(chunkFR.End-1) {
			// All of chunkFR falls within a single hugepage.
			oldSubReclaimed := f.subreclaimed[firstHugeStart]
			incSubReclaimed := chunkFR.Length() / hostarch.PageSize
			newSubReclaimed := oldSubReclaimed + incSubReclaimed
			if newSubReclaimed == pagesPerHugePage {
				// Free this hugepage.
				if oldSubReclaimed != 0 {
					delete(f.subreclaimed, firstHugeStart)
				}
				f.unfreeHuge.RemoveRange(memmap.FileRange{firstHugeStart, firstHugeStart + hostarch.HugePageSize})
			} else {
				f.subreclaimed[firstHugeStart] = newSubReclaimed
			}
			return
		}

		// chunkFR spans at least two hugepages.
		ufseg := f.unfreeHuge.FindSegment(chunkFR.Start)
		if firstHugeStart == chunkFR.Start {
			ufseg = f.unfreeHuge.SplitBefore(ufseg, firstHugeStart)
		} else {
			// chunkFR.Start is not hugepage-aligned. Sub-reclaim the pages between
			// chunkFR.Start and the end of its containing hugepage, then continue
			// with the first following whole hugepage below.
			oldSubReclaimed := f.subreclaimed[firstHugeStart]
			incSubReclaimed := (hostarch.HugePageSize - hostarch.HugePageOffset(chunkFR.Start)) / hostarch.HugePageSize
			newSubReclaimed := oldSubReclaimed + incSubReclaimed
			if newSubReclaimed == pagesPerHugePage {
				// This hugepage will be freed.
				delete(f.subreclaimed, firstHugeStart)
				ufseg = f.unfreeHuge.SplitBefore(ufseg, firstHugeStart)
			} else {
				// This hugepage will not be freed; start with the next one.
				f.subreclaimed[firstHugeStart] = newSubReclaimed
				ufseg = f.unfreeHuge.SplitAfter(ufseg, firstHugeStart+hostarch.HugePageSize).NextSegment()
			}
		}
		// Invariant: ufseg, which is being freed, is split from its predecessor,
		// which is not.

		// Handle whole hugepages in segments before (not containing) the end of
		// the last whole hugepage spanned by chunkFR.
		lastWholeHugeEnd := hostarch.HugePageRoundDown(chunkFR.End)
		for ufseg.Ok() && ufseg.End() <= lastWholeHugeEnd {
			ufseg = f.unfreeHuge.Remove(ufseg).NextSegment()
		}
		// Invariant: If ufseg.Ok(), then it is being freed, and
		// ufseg.Range().Contains(lastWholeHugeEnd).

		if lastWholeHugeEnd == chunkFR.End {
			if ufseg.Ok() && ufseg.Start() < lastWholeHugeEnd {
				// ufseg spans pages both before and after lastWholeHugeEnd; split it
				// and free the first half.
				ufseg = f.unfreeHuge.SplitAfter(ufseg, lastWholeHugeEnd)
				f.unfreeHuge.Remove(ufseg)
			}
		} else {
			// chunkFR.End is not hugepage-aligned. Sub-reclaim the pages between the
			// start of its containing hugepage and chunkFR.End. It must be the case
			// that ufseg.Ok() == true, since it must contain chunkFR.End.
			oldSubReclaimed := f.subreclaimed[lastWholeHugeEnd]
			incSubReclaimed := hostarch.HugePageOffset(chunkFR.End) / hostarch.HugePageSize
			newSubReclaimed := oldSubReclaimed + incSubReclaimed
			if newSubReclaimed == pagesPerHugePage {
				// Include this hugepage when freeing ufseg.
				delete(f.subreclaimed, lastWholeHugeEnd)
				ufseg = f.unfreeHuge.SplitAfter(ufseg, lastWholeHugeEnd)
				f.unfreeHuge.Remove(ufseg)
			} else {
				// Do not include this hugepage when freeing ufseg.
				f.subreclaimed[lastWholeHugeEnd] = newSubReclaimed
				if ufseg.Start() < lastWholeHugeEnd {
					ufseg = f.unfreeHuge.SplitAfter(ufseg, lastWholeHugeEnd)
					f.unfreeHuge.Remove(ufseg)
				}
			}
		}
	})
}

// Decommit uncommits the given pages, causing them to become zeroed.
//
// Preconditions:
//   - fr.Start and fr.End must be page-aligned.
//   - fr.Length() > 0.
//   - At least one reference must be held on all pages in fr.
func (f *MemoryFile) Decommit(fr memmap.FileRange) {
	f.decommitFile(fr)

	f.mu.Lock()
	defer f.mu.Unlock()
	var (
		havekcseg = false
		kcseg     commitmentIterator
	)
	f.forEachChunk(fr, func(chunk *chunkInfo, chunkFR memmap.FileRange) {
		unfree := &f.unfreeSmall
		if chunk.huge {
			unfree = &f.unfreeHuge
		}
		ufseg := unfree.LowerBoundSegmentSplitBefore(chunkFR.Start)
		for ufseg.Ok() && ufseg.Start() < chunkFR.End {
			ufseg = unfree.SplitAfter(ufseg, chunkFR.End)
			uffr := ufseg.Range()
			uf := ufseg.ValuePtr()
			// These pages are still referenced and thus may become committed again
			// at any time; mark them unknown-commitment for now.
			if !havekcseg {
				kcseg = f.knownCommitment.LowerBoundSegmentSplitBefore(uffr.Start)
			} else {
				// Seek kcseg forward from a previous iteration.
				for kcseg.Ok() && kcseg.End() <= uffr.Start {
					kcseg = kcseg.NextSegment()
				}
			}
			for kcseg.Ok() && kcseg.Start() < uffr.End {
				if !kcseg.ValuePtr().committed {
					panic(fmt.Sprintf("referenced pages %v are known-decommitted", kcseg.Range().Intersect(uffr)))
				}
				kcseg = f.knownCommitment.SplitAfter(kcseg, uffr.End)
				kcfr := kcseg.Range()
				f.knownCommittedBytes -= kcfr.Length()
				usage.MemoryAccounting.Dec(kcfr.Length(), uf.kind)
				kcseg = f.knownCommitment.Remove(kcseg).NextSegment()
			}
			// Invalidate committed pages observed by concurrent calls to
			// updateUsageLocked().
			uf.commitSeq = f.commitSeq
			ufseg = unfree.MergePrev(ufseg).NextSegment()
		}
		unfree.MergePrev(ufseg)
	})
}

// MapInternal implements memmap.File.MapInternal.
func (f *MemoryFile) MapInternal(fr memmap.FileRange, at hostarch.AccessType) (safemem.BlockSeq, error) {
	if !fr.WellFormed() || fr.Length() == 0 {
		panic(fmt.Sprintf("invalid range: %v", fr))
	}
	if at.Execute {
		return safemem.BlockSeq{}, linuxerr.EACCES
	}

	chunks := ((fr.End + chunkMask) / chunkSize) - (fr.Start / chunkSize)
	if chunks == 1 {
		// Avoid an unnecessary slice allocation.
		var seq safemem.BlockSeq
		f.forEachMappingSlice(fr, func(bs []byte) {
			seq = safemem.BlockSeqOf(safemem.BlockFromSafeSlice(bs))
		})
		return seq, nil
	}
	blocks := make([]safemem.Block, 0, chunks)
	f.forEachMappingSlice(fr, func(bs []byte) {
		blocks = append(blocks, safemem.BlockFromSafeSlice(bs))
	})
	return safemem.BlockSeqFromSlice(blocks), nil
}

// forEachMappingSlice invokes fn on a sequence of byte slices that
// collectively map all bytes in fr.
func (f *MemoryFile) forEachMappingSlice(fr memmap.FileRange, fn func([]byte)) {
	f.forEachChunk(fr, func(chunk *chunkInfo, chunkFR memmap.FileRange) {
		fn(chunk.sliceAt(chunkFR))
	})
}

// HasUniqueRef returns true if all pages in the given range have exactly one
// reference. A return value of false is inherently racy, but if the caller
// holds a reference on the given range and is preventing other goroutines from
// copying it, then a return value of true is not racy.
//
// Preconditions: At least one reference must be held on all pages in fr.
func (f *MemoryFile) HasUniqueRef(fr memmap.FileRange) (hasUniqueRef bool) {
	hasUniqueRef = true
	f.mu.Lock()
	defer f.mu.Unlock()
	f.forEachChunk(fr, func(chunk *chunkInfo, chunkFR memmap.FileRange) {
		if !hasUniqueRef {
			return
		}
		unfree := &f.unfreeSmall
		if chunk.huge {
			unfree = &f.unfreeHuge
		}
		for ufseg := unfree.FindSegment(fr.Start); ufseg.Ok() && ufseg.Start() < fr.End; ufseg = ufseg.NextSegment() {
			if ufseg.ValuePtr().refs != 1 {
				hasUniqueRef = false
				return
			}
		}
	})
	return
}

// MarkEvictable allows f to request memory deallocation by calling
// user.Evict(er) in the future.
//
// Redundantly marking an already-evictable range as evictable has no effect.
func (f *MemoryFile) MarkEvictable(user EvictableMemoryUser, er EvictableRange) {
	f.mu.Lock()
	defer f.mu.Unlock()
	info, ok := f.evictable[user]
	if !ok {
		info = &evictableMemoryUserInfo{}
		f.evictable[user] = info
	}
	gap := info.ranges.LowerBoundGap(er.Start)
	for gap.Ok() && gap.Start() < er.End {
		gapER := gap.Range().Intersect(er)
		if gapER.Length() == 0 {
			gap = gap.NextGap()
			continue
		}
		gap = info.ranges.Insert(gap, gapER, evictableRangeSetValue{}).NextGap()
	}
	if !info.evicting {
		switch f.opts.DelayedEviction {
		case DelayedEvictionDisabled:
			// Kick off eviction immediately.
			f.startEvictionGoroutineLocked(user, info)
		case DelayedEvictionEnabled:
			if !f.opts.UseHostMemcgPressure {
				// Ensure that the reclaimer goroutine is running, so that it
				// can start eviction when necessary.
				f.reclaimCond.Signal()
			}
		}
	}
}

// MarkUnevictable informs f that user no longer considers er to be evictable,
// so the MemoryFile should no longer call user.Evict(er). Note that, per
// EvictableMemoryUser.Evict's documentation, user.Evict(er) may still be
// called even after MarkUnevictable returns due to race conditions, and
// implementations of EvictableMemoryUser must handle this possibility.
//
// Redundantly marking an already-unevictable range as unevictable has no
// effect.
func (f *MemoryFile) MarkUnevictable(user EvictableMemoryUser, er EvictableRange) {
	f.mu.Lock()
	defer f.mu.Unlock()
	info, ok := f.evictable[user]
	if !ok {
		return
	}
	seg := info.ranges.LowerBoundSegment(er.Start)
	for seg.Ok() && seg.Start() < er.End {
		seg = info.ranges.Isolate(seg, er)
		seg = info.ranges.Remove(seg).NextSegment()
	}
	// We can only remove info if there's no eviction goroutine running on its
	// behalf.
	if !info.evicting && info.ranges.IsEmpty() {
		delete(f.evictable, user)
	}
}

// MarkAllUnevictable informs f that user no longer considers any offsets to be
// evictable. It otherwise has the same semantics as MarkUnevictable.
func (f *MemoryFile) MarkAllUnevictable(user EvictableMemoryUser) {
	f.mu.Lock()
	defer f.mu.Unlock()
	info, ok := f.evictable[user]
	if !ok {
		return
	}
	info.ranges.RemoveAll()
	// We can only remove info if there's no eviction goroutine running on its
	// behalf.
	if !info.evicting {
		delete(f.evictable, user)
	}
}

// ShouldCacheEvictable returns true if f is meaningfully delaying evictions of
// evictable memory, such that it may be advantageous to cache data in
// evictable memory. The value returned by ShouldCacheEvictable may change
// between calls.
func (f *MemoryFile) ShouldCacheEvictable() bool {
	return f.opts.DelayedEviction == DelayedEvictionManual || f.opts.UseHostMemcgPressure
}

// UpdateUsage ensures that the memory usage statistics in
// usage.MemoryAccounting are up to date.
func (f *MemoryFile) UpdateUsage() error {
	f.mu.Lock()
	defer f.mu.Unlock()

	// If we already know of every committed page, skip scanning.
	currentUsage, err := f.TotalUsage()
	if err != nil {
		return err
	}
	if currentUsage == f.knownCommittedBytes {
		return nil
	}

	// Linux updates usage values at CONFIG_HZ; throttle our scans to the same
	// frequency.
	now := time.Now()
	if now.Before(f.nextCommitScan) {
		return nil
	}
	f.nextCommitScan = now.Add(time.Second / linux.CLOCKS_PER_SEC)

	return f.updateUsageLocked(mincore)
}

// updateUsageLocked attempts to detect commitment of unknown-commitment
// pages by invoking checkCommitted, which is a function that, for each page i
// in bs, sets committed[i] to 1 if the page is committed and 0 otherwise.
//
// Precondition: f.mu must be held; it may be unlocked and reacquired.
// +checklocks:f.mu
func (f *MemoryFile) updateUsageLocked(checkCommitted func(bs []byte, committed []byte) error) error {
	// Reused mincore buffer.
	var buf []byte

	scanStart := uint64(0)
	for {
		kcgap := f.knownCommitment.LowerBoundLargeEnoughGap(scanStart, 1)
		if !kcgap.Ok() {
			break
		}
		fr := kcgap.Range()
		var checkErr error
		f.forEachChunk(fr, func(chunk *chunkInfo, chunkFR memmap.FileRange) {
			if checkErr != nil {
				return
			}

			s := chunk.sliceAt(chunkFR)

			// Ensure that we have sufficient buffer for the call (one byte per
			// page). The length of each slice must be page-aligned.
			bufLen := len(s) / hostarch.PageSize
			if len(buf) < bufLen {
				buf = make([]byte, bufLen)
			}

			// Query for new pages in core.
			// NOTE(b/165896008): mincore (which is passed as checkCommitted by
			// f.UpdateUsage()) might take a really long time. So unlock f.mu while
			// checkCommitted runs.
			lastCommitSeq := f.commitSeq
			f.commitSeq++
			f.mu.Unlock() // +checklocksforce
			err := checkCommitted(s, buf)
			f.mu.Lock()
			if err != nil {
				checkErr = err
				return
			}

			// Reconcile internal state with buf. Since we temporarily dropped f.mu,
			// f.knownCommitment may have changed, and kcgap is no longer valid.
			unfree := &f.unfreeSmall
			if chunk.huge {
				unfree = &f.unfreeHuge
			}
			kcgap := f.knownCommitment.LowerBoundLargeEnoughGap(chunkFR.Start, 1)
			for kcgap.Ok() && kcgap.Start() < chunkFR.End {
				// kcgap represents a range of pages whose commitment is still
				// unknown that overlaps the range of pages in this chunk that
				// we scanned. For the pages in the intersection of these two
				// ranges, determine which of these pages are now known to be
				// committed.
				kcgFR := kcgap.Range().Intersect(chunkFR)
				i := (kcgFR.Start - chunkFR.Start) / hostarch.PageSize
				end := (kcgFR.End - chunkFR.Start) / hostarch.PageSize
				for i < end {
					if buf[i]&0x1 == 0 {
						// Unknown-commitment pages that are currently uncommitted could
						// still become committed at any time, so we can't mark them
						// known-decommitted; leave them marked unknown-commitment until
						// the next scan.
						i++
						continue
					}
					// Scan to the end of this committed range.
					j := i + 1
					for ; j < end; j++ {
						if buf[j]&0x1 == 0 {
							break
						}
					}
					commitFR := memmap.FileRange{
						Start: chunkFR.Start + (i * hostarch.PageSize),
						End:   chunkFR.Start + (j * hostarch.PageSize),
					}
					// We need to iterate unfree to check commitSeq and determine
					// accounting kind. Free pages are known-decommitted, so there must
					// be segments in unfree spanning all of commitFR.
					ufseg := unfree.FindSegment(commitFR.Start)
					for ufseg.Start() < commitFR.End {
						uf := ufseg.ValuePtr()
						// If uf.commitSeq > lastCommitSeq, then uf may have
						// been decommitted, either via Decommit() or as a
						// result of being freed, concurrently with or after
						// checkCommitted, so we cannot conclude that these
						// pages are known-committed. Leave them
						// unknown-commitment, to be scanned by the next call
						// to updateUsageLocked().
						if uf.commitSeq <= lastCommitSeq {
							kcFR := ufseg.Range().Intersect(commitFR)
							kcgap = f.knownCommitment.Insert(kcgap, kcFR, commitmentInfo{
								committed: true,
							}).NextGap()
							f.knownCommittedBytes += kcFR.Length()
							usage.MemoryAccounting.Inc(kcFR.Length(), uf.kind)
						}
						ufseg = ufseg.NextSegment()
					}
					i = j
				}
				kcgap = kcgap.NextLargeEnoughGap(1)
			}
		})
		if checkErr != nil {
			return checkErr
		}
		scanStart = fr.End
	}

	return nil
}

// TotalUsage returns an aggregate usage for all memory statistics except
// Mapped (which is external to MemoryFile). This is generally much cheaper
// than UpdateUsage, but will not provide a fine-grained breakdown.
func (f *MemoryFile) TotalUsage() (uint64, error) {
	// Stat the underlying file to discover the underlying usage. stat(2)
	// always reports the allocated block count in units of 512 bytes. This
	// includes pages in the page cache and swapped pages.
	var stat unix.Stat_t
	if err := unix.Fstat(int(f.file.Fd()), &stat); err != nil {
		return 0, err
	}
	return uint64(stat.Blocks * 512), nil
}

// TotalSize returns the current size of the backing file in bytes, which is an
// upper bound on the amount of memory that can currently be allocated from the
// MemoryFile. The value returned by TotalSize is permitted to change.
func (f *MemoryFile) TotalSize() uint64 {
	return uint64(len(f.chunksLoad())) * chunkSize
}

// File returns the backing file.
func (f *MemoryFile) File() *os.File {
	return f.file
}

// FD implements memmap.File.FD.
func (f *MemoryFile) FD() int {
	return int(f.file.Fd())
}

// HugepagesEnabled returns true if the MemoryFile expects to back allocations
// for which AllocOpts.Hugepage == true with huge pages.
func (f *MemoryFile) HugepagesEnabled() bool {
	return f.opts.ExpectHugepages
}

// String implements fmt.Stringer.String.
//
// Note that because f.String locks f.mu, calling f.String internally
// (including indirectly through the fmt package) risks recursive locking.
// Within the pgalloc package, use f.usage directly instead.
func (f *MemoryFile) String() string {
	return "FIXME(jamieliu)"
}

// StartEvictions requests that f evict all evictable allocations. It does not
// wait for eviction to complete; for this, see MemoryFile.WaitForEvictions.
func (f *MemoryFile) StartEvictions() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.startEvictionsLocked()
}

// Preconditions: f.mu must be locked.
func (f *MemoryFile) startEvictionsLocked() bool {
	startedAny := false
	for user, info := range f.evictable {
		// Don't start multiple goroutines to evict the same user's
		// allocations.
		if !info.evicting {
			f.startEvictionGoroutineLocked(user, info)
			startedAny = true
		}
	}
	return startedAny
}

// Preconditions:
//   - info == f.evictable[user].
//   - !info.evicting.
//   - f.mu must be locked.
func (f *MemoryFile) startEvictionGoroutineLocked(user EvictableMemoryUser, info *evictableMemoryUserInfo) {
	info.evicting = true
	f.evictionWG.Add(1)
	go func() { // S/R-SAFE: f.evictionWG
		defer f.evictionWG.Done()
		for {
			f.mu.Lock()
			info, ok := f.evictable[user]
			if !ok {
				// This shouldn't happen: only this goroutine is permitted
				// to delete this entry.
				f.mu.Unlock()
				panic(fmt.Sprintf("evictableMemoryUserInfo for EvictableMemoryUser %v deleted while eviction goroutine running", user))
			}
			if info.ranges.IsEmpty() {
				delete(f.evictable, user)
				f.mu.Unlock()
				return
			}
			// Evict from the end of info.ranges, under the assumption that
			// if ranges in user start being used again (and are
			// consequently marked unevictable), such uses are more likely
			// to start from the beginning of user.
			seg := info.ranges.LastSegment()
			er := seg.Range()
			info.ranges.Remove(seg)
			// user.Evict() must be called without holding f.mu to avoid
			// circular lock ordering.
			f.mu.Unlock()
			user.Evict(context.Background(), er)
		}
	}()
}

// WaitForEvictions blocks until f is no longer evicting any evictable
// allocations.
func (f *MemoryFile) WaitForEvictions() {
	f.evictionWG.Wait()
}

type unwasteSetFunctions struct{}

func (unwasteSetFunctions) MinKey() uint64 {
	return 0
}

func (unwasteSetFunctions) MaxKey() uint64 {
	return math.MaxUint64
}

func (unwasteSetFunctions) ClearValue(val *unwasteInfo) {
}

func (unwasteSetFunctions) Merge(_ memmap.FileRange, val1 unwasteInfo, _ memmap.FileRange, val2 unwasteInfo) (unwasteInfo, bool) {
	return val1, val1 == val2
}

func (unwasteSetFunctions) Split(_ memmap.FileRange, val unwasteInfo, _ uint64) (unwasteInfo, unwasteInfo) {
	return val, val
}

type unfreeSetFunctions struct{}

func (unfreeSetFunctions) MinKey() uint64 {
	return 0
}

func (unfreeSetFunctions) MaxKey() uint64 {
	return math.MaxUint64
}

func (unfreeSetFunctions) ClearValue(val *unfreeInfo) {
}

func (unfreeSetFunctions) Merge(_ memmap.FileRange, val1 unfreeInfo, _ memmap.FileRange, val2 unfreeInfo) (unfreeInfo, bool) {
	return val1, val1 == val2
}

func (unfreeSetFunctions) Split(_ memmap.FileRange, val unfreeInfo, _ uint64) (unfreeInfo, unfreeInfo) {
	return val, val
}

type commitmentSetFunctions struct{}

func (commitmentSetFunctions) MinKey() uint64 {
	return 0
}

func (commitmentSetFunctions) MaxKey() uint64 {
	return math.MaxUint64
}

func (commitmentSetFunctions) ClearValue(val *commitmentInfo) {
}

func (commitmentSetFunctions) Merge(_ memmap.FileRange, val1 commitmentInfo, _ memmap.FileRange, val2 commitmentInfo) (commitmentInfo, bool) {
	return val1, val1 == val2
}

func (commitmentSetFunctions) Split(_ memmap.FileRange, val commitmentInfo, _ uint64) (commitmentInfo, commitmentInfo) {
	return val, val
}

// evictableRangeSetValue is the value type of evictableRangeSet.
type evictableRangeSetValue struct{}

type evictableRangeSetFunctions struct{}

func (evictableRangeSetFunctions) MinKey() uint64 {
	return 0
}

func (evictableRangeSetFunctions) MaxKey() uint64 {
	return math.MaxUint64
}

func (evictableRangeSetFunctions) ClearValue(val *evictableRangeSetValue) {
}

func (evictableRangeSetFunctions) Merge(_ EvictableRange, _ evictableRangeSetValue, _ EvictableRange, _ evictableRangeSetValue) (evictableRangeSetValue, bool) {
	return evictableRangeSetValue{}, true
}

func (evictableRangeSetFunctions) Split(_ EvictableRange, _ evictableRangeSetValue, _ uint64) (evictableRangeSetValue, evictableRangeSetValue) {
	return evictableRangeSetValue{}, evictableRangeSetValue{}
}
