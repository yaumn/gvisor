load("//tools:defs.bzl", "go_library", "go_test")
load("//tools/go_generics:defs.bzl", "go_template_instance")
load("//pkg/sync/locking:locking.bzl", "declare_mutex")

package(
    default_applicable_licenses = ["//:license"],
    licenses = ["notice"],
)

declare_mutex(
    name = "mappings_mutex",
    out = "mappings_mutex.go",
    package = "pgalloc",
    prefix = "mappings",
)

declare_mutex(
    name = "memory_file_mutex",
    out = "memory_file_mutex.go",
    package = "pgalloc",
    prefix = "memoryFile",
)

go_template_instance(
    name = "evictable_range",
    out = "evictable_range.go",
    package = "pgalloc",
    prefix = "Evictable",
    template = "//pkg/segment:generic_range",
    types = {
        "T": "uint64",
    },
)

go_template_instance(
    name = "evictable_range_set",
    out = "evictable_range_set.go",
    package = "pgalloc",
    prefix = "evictableRange",
    template = "//pkg/segment:generic_set",
    types = {
        "Key": "uint64",
        "Range": "EvictableRange",
        "Value": "evictableRangeSetValue",
        "Functions": "evictableRangeSetFunctions",
    },
)

go_template_instance(
    name = "commitment_set",
    out = "commitment_set.go",
    consts = {
        "minDegree": "10",
        "trackGaps": "1",
    },
    imports = {
        "memmap": "gvisor.dev/gvisor/pkg/sentry/memmap",
    },
    package = "pgalloc",
    prefix = "commitment",
    template = "//pkg/segment:generic_set",
    types = {
        "Key": "uint64",
        "Range": "memmap.FileRange",
        "Value": "commitmentInfo",
        "Functions": "commitmentSetFunctions",
    },
)

go_template_instance(
    name = "unfree_set",
    out = "unfree_set.go",
    consts = {
        "minDegree": "10",
        "trackGaps": "1",
    },
    imports = {
        "memmap": "gvisor.dev/gvisor/pkg/sentry/memmap",
    },
    package = "pgalloc",
    prefix = "unfree",
    template = "//pkg/segment:generic_set",
    types = {
        "Key": "uint64",
        "Range": "memmap.FileRange",
        "Value": "unfreeInfo",
        "Functions": "unfreeSetFunctions",
    },
)

go_template_instance(
    name = "unwaste_set",
    out = "unwaste_set.go",
    consts = {
        "minDegree": "10",
        "trackGaps": "1",
    },
    imports = {
        "memmap": "gvisor.dev/gvisor/pkg/sentry/memmap",
    },
    package = "pgalloc",
    prefix = "unwaste",
    template = "//pkg/segment:generic_set",
    types = {
        "Key": "uint64",
        "Range": "memmap.FileRange",
        "Value": "unwasteInfo",
        "Functions": "unwasteSetFunctions",
    },
)

go_template_instance(
    name = "seqatomic_chunkinfoslice",
    out = "seqatomic_chunkinfoslice_unsafe.go",
    package = "pgalloc",
    suffix = "ChunkInfoSlice",
    template = "//pkg/sync/seqatomic:generic_seqatomic",
    types = {
        "Value": "[]chunkInfo",
    },
)

go_library(
    name = "pgalloc",
    srcs = [
        "commitment_set.go",
        "context.go",
        "evictable_range.go",
        "evictable_range_set.go",
        "mappings_mutex.go",
        "memory_file_mutex.go",
        "pgalloc.go",
        "pgalloc_unsafe.go",
        "save_restore.go",
        "seqatomic_chunkinfoslice_unsafe.go",
        "unfree_set.go",
        "unwaste_set.go",
    ],
    visibility = ["//pkg/sentry:internal"],
    deps = [
        "//pkg/abi/linux",
        "//pkg/atomicbitops",
        "//pkg/context",
        "//pkg/errors/linuxerr",
        "//pkg/gohacks",
        "//pkg/hostarch",
        "//pkg/log",
        "//pkg/memutil",
        "//pkg/safemem",
        "//pkg/sentry/arch",
        "//pkg/sentry/hostmm",
        "//pkg/sentry/memmap",
        "//pkg/sentry/usage",
        "//pkg/state",
        "//pkg/state/wire",
        "//pkg/sync",
        "//pkg/sync/locking",
        "//pkg/usermem",
        "@org_golang_x_sys//unix:go_default_library",
    ],
)

go_test(
    name = "pgalloc_test",
    size = "small",
    srcs = ["pgalloc_test.go"],
    library = ":pgalloc",
    deps = ["//pkg/hostarch"],
)
