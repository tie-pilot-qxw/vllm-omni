# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class TaskKind(str, Enum):
    TEXT_ENCODE = "text_encode"
    DIT_PREPARE = "dit_prepare"
    TIMESTEP_PREPARE = "timestep_prepare"
    DIT_STEP_CHUNK = "dit_step_chunk"
    VAE_DECODE = "vae_decode"
    FINALIZE = "finalize"
    # RESHARD is a cross-group artifact migration task. It is a collective
    # operation spanning src∪dst ranks, so it does not belong to any single
    # execution group. Tasks of this kind must have group_id=None and must
    # carry src/dst group ids in payload (see RESHARD_PAYLOAD_*).
    RESHARD = "reshard"


# Payload key conventions for RESHARD tasks. The compiler sets these and the
# scheduler reads them when bypassing normal worker dispatch.
RESHARD_PAYLOAD_SRC_GROUP_ID = "src_group_id"
RESHARD_PAYLOAD_DST_GROUP_ID = "dst_group_id"


class ArtifactKind(str, Enum):
    REQUEST_STATE = "request_state"
    TENSOR = "tensor"
    OUTPUT = "output"


class ArtifactLayout(str, Enum):
    HOST = "host"
    WORKER_LOCAL = "worker_local"


class TensorLayoutKind(str, Enum):
    """Logical tensor placement within an execution group.

    These layouts describe what each group rank owns, not how ranks should
    communicate during migration. Concrete P2P send/recv pairs are derived
    later from src/dst group membership plus src/dst layouts.
    """

    OWNER_ONLY = "owner_only"
    REPLICATED = "replicated"
    SHARDED = "sharded"


@dataclass(frozen=True)
class TensorShardSpec:
    dim: int

    def __post_init__(self) -> None:
        if self.dim < 0:
            raise ValueError(f"shard dim must be >= 0, got {self.dim}")


@dataclass(frozen=True)
class TensorLayout:
    kind: TensorLayoutKind
    shard: TensorShardSpec | None = None

    def __post_init__(self) -> None:
        kind = TensorLayoutKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if kind == TensorLayoutKind.SHARDED:
            if self.shard is None:
                raise ValueError("sharded tensor layout requires a shard spec")
            return
        if self.shard is not None:
            raise ValueError(f"{kind.value} tensor layout must not set a shard spec")


@dataclass(frozen=True)
class TensorFieldLayout:
    field_path: tuple[str, ...]
    local_shape: tuple[int, ...]
    dtype: str
    layout: TensorLayout
    global_shape: tuple[int, ...] | None = None
    global_offset: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.field_path:
            raise ValueError("tensor field layout requires a path")
        object.__setattr__(self, "field_path", tuple(str(part) for part in self.field_path))
        object.__setattr__(self, "local_shape", tuple(int(dim) for dim in self.local_shape))
        if any(dim < 0 for dim in self.local_shape):
            raise ValueError(f"tensor field local_shape must be non-negative, got {self.local_shape}")
        if not self.dtype:
            raise ValueError("tensor field layout requires dtype")
        global_shape = self.global_shape
        if global_shape is None:
            global_shape = self.local_shape
        global_offset = self.global_offset
        if global_offset is None:
            global_offset = tuple(0 for _ in global_shape)
        object.__setattr__(self, "global_shape", tuple(int(dim) for dim in global_shape))
        object.__setattr__(self, "global_offset", tuple(int(dim) for dim in global_offset))
        if len(self.local_shape) != len(self.global_shape):
            raise ValueError(
                "tensor field local/global rank mismatch: "
                f"local_shape={self.local_shape}, global_shape={self.global_shape}"
            )
        if len(self.global_offset) != len(self.global_shape):
            raise ValueError(
                "tensor field offset/global rank mismatch: "
                f"global_offset={self.global_offset}, global_shape={self.global_shape}"
            )
        if any(dim < 0 for dim in self.global_shape):
            raise ValueError(f"tensor field global_shape must be non-negative, got {self.global_shape}")
        if any(dim < 0 for dim in self.global_offset):
            raise ValueError(f"tensor field global_offset must be non-negative, got {self.global_offset}")
        for offset, local, full in zip(self.global_offset, self.local_shape, self.global_shape, strict=True):
            if offset + local > full:
                raise ValueError(
                    "tensor field local slice exceeds global shape: "
                    f"offset={self.global_offset}, local_shape={self.local_shape}, global_shape={self.global_shape}"
                )


@dataclass(frozen=True)
class OutputArtifactLayout:
    handle: "ArtifactHandle"
    tensors: tuple[TensorFieldLayout, ...]

    def __post_init__(self) -> None:
        if not self.handle.codec_id:
            raise ValueError("output artifact layout requires handle.codec_id")
        object.__setattr__(self, "tensors", tuple(self.tensors))


@dataclass(frozen=True)
class TensorSliceSpec:
    field_path: tuple[str, ...]
    offset: tuple[int, ...]
    shape: tuple[int, ...]
    dtype: str

    def __post_init__(self) -> None:
        if not self.field_path:
            raise ValueError("tensor slice spec requires field_path")
        object.__setattr__(self, "field_path", tuple(str(part) for part in self.field_path))
        object.__setattr__(self, "offset", tuple(int(dim) for dim in self.offset))
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        if len(self.offset) != len(self.shape):
            raise ValueError(
                f"tensor slice offset/shape rank mismatch: offset={self.offset}, shape={self.shape}"
            )
        if any(dim < 0 for dim in self.offset):
            raise ValueError(f"tensor slice offset must be non-negative, got {self.offset}")
        if any(dim < 0 for dim in self.shape):
            raise ValueError(f"tensor slice shape must be non-negative, got {self.shape}")
        if not self.dtype:
            raise ValueError("tensor slice spec requires dtype")


@dataclass(frozen=True)
class TensorMigrationEdge:
    artifact_id: str
    src_rank: int
    dst_rank: int
    src_slice: TensorSliceSpec
    dst_slice: TensorSliceSpec
    local_copy: bool = False

    def __post_init__(self) -> None:
        if not self.artifact_id:
            raise ValueError("tensor migration edge requires artifact_id")
        if self.src_slice.dtype != self.dst_slice.dtype:
            raise ValueError(
                "tensor migration edge dtype mismatch: "
                f"src={self.src_slice.dtype}, dst={self.dst_slice.dtype}"
            )
        if self.src_slice.shape != self.dst_slice.shape:
            raise ValueError(
                "tensor migration edge slice shape mismatch: "
                f"src={self.src_slice.shape}, dst={self.dst_slice.shape}"
            )


@dataclass(frozen=True)
class ArtifactMigrationSchedule:
    migrate_id: str
    request_id: str
    src_group_id: str
    dst_group_id: str
    src_ranks: tuple[int, ...]
    dst_ranks: tuple[int, ...]
    participant_ranks: tuple[int, ...]
    edges: tuple[TensorMigrationEdge, ...]

    def __post_init__(self) -> None:
        if not self.migrate_id:
            raise ValueError("artifact migration schedule requires migrate_id")
        if not self.request_id:
            raise ValueError("artifact migration schedule requires request_id")
        object.__setattr__(self, "src_ranks", tuple(int(rank) for rank in self.src_ranks))
        object.__setattr__(self, "dst_ranks", tuple(int(rank) for rank in self.dst_ranks))
        object.__setattr__(self, "participant_ranks", tuple(int(rank) for rank in self.participant_ranks))
        object.__setattr__(self, "edges", tuple(self.edges))


class TaskStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    DISPATCHED = "dispatched"
    RUNNING = "running"
    FINISHED = "finished"
    FAILED = "failed"


class WorkerEventKind(str, Enum):
    TASK_LAUNCH_BEGIN = "task_launch_begin"
    TASK_LAUNCH_END = "task_launch_end"
    TASK_EXEC_BEGIN = "task_exec_begin"
    TASK_EXEC_END = "task_exec_end"
    TASK_FAILED = "task_failed"
    REQUEST_FINISHED = "request_finished"
    REQUEST_FAILED = "request_failed"


@dataclass(frozen=True)
class ParallelSpec:
    """Static parallel metadata for scheduling and fixed worker sessions."""

    tp: int = 1
    sp: int = 1
    cfg: int = 1
    cfg_parallel: bool = False

    def __post_init__(self) -> None:
        if self.tp < 1:
            raise ValueError(f"tp must be >= 1, got {self.tp}")
        if self.sp < 1:
            raise ValueError(f"sp must be >= 1, got {self.sp}")
        cfg = int(self.cfg)
        if cfg < 1:
            raise ValueError(f"cfg must be >= 1, got {cfg}")
        if self.cfg_parallel and cfg == 1:
            cfg = 2
        object.__setattr__(self, "cfg", cfg)
        object.__setattr__(self, "cfg_parallel", cfg > 1)


@dataclass(frozen=True)
class ExecutionGroupSpec:
    group_id: str
    ranks: tuple[int, ...]
    parallel_spec: ParallelSpec
    supported_task_kinds: tuple[TaskKind, ...]
    ulysses_degree: int = 1
    ring_degree: int = 1

    def __post_init__(self) -> None:
        if not self.ranks:
            raise ValueError("execution group must contain at least one rank")
        ulysses_degree = int(self.ulysses_degree)
        ring_degree = int(self.ring_degree)
        sp = int(self.parallel_spec.sp)
        if ulysses_degree < 1:
            raise ValueError(f"ulysses_degree must be >= 1, got {self.ulysses_degree}")
        if ring_degree < 1:
            raise ValueError(f"ring_degree must be >= 1, got {self.ring_degree}")
        if sp != ulysses_degree * ring_degree and ulysses_degree == 1 and ring_degree == 1:
            ulysses_degree = sp
            object.__setattr__(self, "ulysses_degree", ulysses_degree)
        if sp != ulysses_degree * ring_degree:
            raise ValueError(
                "execution group sequence parallel mismatch: "
                f"sp={self.parallel_spec.sp} != ulysses_degree({ulysses_degree}) * ring_degree({ring_degree})"
            )


@dataclass(frozen=True)
class ArtifactHandle:
    request_id: str
    artifact_id: str
    kind: ArtifactKind
    layout: ArtifactLayout
    producer_task_id: str | None = None
    codec_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArtifactValue:
    handle: ArtifactHandle
    value: Any


@dataclass(frozen=True)
class WorkerLocalArtifactRef:
    handle: ArtifactHandle
    group_id: str
    worker_rank: int


@dataclass(frozen=True)
class MigrateArtifactSpec:
    """One artifact in a MigrateArtifactsPlan.

    `handle` is the *destination* artifact handle: the id under which the
    migrated value will be registered in the dst group's local artifact store.

    `source_handle` is the *source* handle on the src group. It is optional;
    when omitted, the source and destination share the same artifact_id (the
    legacy single-id behavior). When the compiler renames the artifact across
    the boundary (e.g. `state_text` -> `state_text_dit`), `source_handle` must
    be set so the migration path can locate the source value on src ranks and
    can call the codec with the correct artifact_id (codecs sometimes derive
    the layout from the artifact_id itself).
    """

    handle: ArtifactHandle
    source_handle: ArtifactHandle | None = None

    def __post_init__(self) -> None:
        if not self.handle.artifact_id:
            raise ValueError("migrate artifact spec requires artifact_id")
        if self.source_handle is not None:
            if not self.source_handle.artifact_id:
                raise ValueError("migrate artifact spec source_handle requires artifact_id")
            if self.source_handle.request_id != self.handle.request_id:
                raise ValueError(
                    "migrate artifact spec source/destination request_id mismatch: "
                    f"src={self.source_handle.request_id!r}, dst={self.handle.request_id!r}"
                )

    @property
    def effective_source_handle(self) -> ArtifactHandle:
        return self.source_handle if self.source_handle is not None else self.handle


@dataclass(frozen=True)
class MigrateArtifactsPlan:
    request_id: str
    src_group_id: str
    dst_group_id: str
    artifacts: tuple[MigrateArtifactSpec, ...]
    request_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("migrate artifacts plan requires request_id")
        if not self.src_group_id:
            raise ValueError("migrate artifacts plan requires src_group_id")
        if not self.dst_group_id:
            raise ValueError("migrate artifacts plan requires dst_group_id")
        if not self.artifacts:
            raise ValueError("migrate artifacts plan requires at least one artifact spec")
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "request_metadata", dict(self.request_metadata))
        for artifact in self.artifacts:
            if artifact.handle.request_id != self.request_id:
                raise ValueError(
                    "migrate artifacts plan request_id mismatch: "
                    f"plan={self.request_id}, artifact={artifact.handle.request_id}"
                )


@dataclass(frozen=True)
class StepRange:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(f"start must be >= 0, got {self.start}")
        if self.end <= self.start:
            raise ValueError(f"end must be greater than start, got start={self.start}, end={self.end}")


@dataclass
class InferenceTask:
    task_id: str
    request_id: str
    kind: TaskKind
    group_id: str | None
    parallel_spec: ParallelSpec
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    dependencies: tuple[str, ...] = ()
    inputs: tuple[ArtifactHandle, ...] = ()
    outputs: tuple[ArtifactHandle, ...] = ()
    step_range: StepRange | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class RequestExecutionPlan:
    request_id: str
    tasks: dict[str, InferenceTask]
    terminal_task_ids: tuple[str, ...]
    initial_artifacts: tuple[ArtifactValue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class WorkerEvent:
    event_id: str
    task_id: str
    request_id: str
    group_id: str
    worker_rank: int
    kind: WorkerEventKind
    timestamp_ns: int
    message: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
