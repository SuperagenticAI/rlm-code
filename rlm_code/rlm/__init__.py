"""
RLM runtime for RLM Code CLI.

This module provides the core RLM (Recursive Language Model) runtime,
implementing the paradigm from "Recursive Language Models" (2025).

Key components:
- PureRLMEnvironment: Paper-compliant RLM with context-as-variable
- REPLVariable/REPLHistory: Structured REPL state management
- FINAL/FINAL_VAR: Clean termination patterns
- RLMRunner: Orchestrates RLM execution loops
- ParadigmComparator: Side-by-side comparison of RLM paradigms
- MemoryCompactor: Context window management via summarization
"""

# Approval / HITL Gates
from .approval import (
    ApprovalAuditLog,
    ApprovalGate,
    ApprovalHandler,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    AuditEntry,
    AutoApproveHandler,
    AutoDenyHandler,
    CallbackApprovalHandler,
    ConsoleApprovalHandler,
    RiskAssessment,
    RiskAssessor,
    ToolRiskLevel,
)
from .code_interpreter import CodeInterpreter, CodeResult, LocalInterpreter
from .comparison import (
    ComparisonResult,
    Paradigm,
    ParadigmComparator,
    ParadigmResult,
    create_comparison_report,
)
from .config_schema import (
    BenchmarkConfig,
    MCPServerConfig,
    RLMConfig,
    SandboxConfig,
    TrajectoryConfig,
    generate_sample_config,
    get_default_config,
)
from .docker_interpreter import DockerPersistentInterpreter
from .environments import (
    DSPyCodingRLMEnvironment,
    EnvironmentActionResult,
    EnvironmentDoctorCheck,
    GenericRLMEnvironment,
    RLMRewardProfile,
)
from .events import (
    RLMEventBus,
    RLMEventCollector,
    RLMEventData,
    RLMEventType,
    RLMRuntimeEvent,
)
from .frameworks import (
    ADKRLMFrameworkAdapter,
    DeepAgentsFrameworkAdapter,
    DSPyRLMFrameworkAdapter,
    FrameworkAdapterRegistry,
    FrameworkEpisodeResult,
    FrameworkStepRecord,
    GoogleADKFrameworkAdapter,
    PydanticAIFrameworkAdapter,
)
from .leaderboard import (
    Leaderboard,
    LeaderboardEntry,
    LeaderboardFilter,
    RankingMetric,
    RankingResult,
    SortOrder,
    aggregate_by_field,
    compute_trend,
    leaderboard_cli,
)
from .memory_compaction import (
    CompactionConfig,
    CompactionResult,
    ConversationMemory,
    MemoryCompactor,
)
from .mock_interpreter import MockInterpreter
from .monty_interpreter import (
    MontyCodeResult,
    MontyCodeValidator,
    MontyExecutionStats,
    MontyInterpreter,
    create_rlm_monty_interpreter,
)
from .observability import (
    LocalJSONLSink,
    MLflowSink,
    RLMObservability,
    RLMObservabilitySink,
)
from .observability_sinks import (
    CompositeSink,
    LangFuseSink,
    LangSmithSink,
    LogfireSink,
    OpenTelemetrySink,
    create_all_sinks_from_env,
    create_langfuse_sink_from_env,
    create_langsmith_sink_from_env,
    create_logfire_sink_from_env,
    create_otel_sink_from_env,
)

# Policy Lab - Hot-swappable policies
from .policies import (
    ActionSelectionPolicy,
    BeamSearchActionPolicy,
    CompactionPolicy,
    ConfidenceTerminationPolicy,
    # Reward policies
    DefaultRewardPolicy,
    DeterministicCompactionPolicy,
    # Termination policies
    FinalPatternTerminationPolicy,
    # Action policies
    GreedyActionPolicy,
    HierarchicalCompactionPolicy,
    LenientRewardPolicy,
    # Compaction policies
    LLMCompactionPolicy,
    MCTSActionPolicy,
    Policy,
    PolicyRegistry,
    ResearchRewardPolicy,
    RewardPolicy,
    RewardThresholdTerminationPolicy,
    SamplingActionPolicy,
    SlidingWindowCompactionPolicy,
    StrictRewardPolicy,
    TerminationPolicy,
)
from .pure_rlm_environment import PureRLMConfig, PureRLMEnvironment
from .repl_types import (
    ImmutableHistory,
    ImmutableHistoryEntry,
    REPLEntry,
    REPLHistory,
    REPLResult,
    REPLVariable,
)
from .runner import (
    RLMBenchmarkComparison,
    RLMBenchmarkReport,
    RLMBenchmarkResult,
    RLMJudgeResult,
    RLMRunner,
    RLMRunResult,
)
from .session_replay import (
    SessionComparison,
    SessionEvent,
    SessionEventType,
    SessionRecorder,
    SessionReplayer,
    SessionSnapshot,
    SessionStore,
    StepState,
    compare_sessions,
    create_recorder,
    load_session,
)
from .task_signature import TaskSignature
from .termination import (
    FINAL,
    FINAL_VAR,
    SUBMIT,
    FinalDetection,
    FinalOutput,
    SubmitOutput,
    detect_final_in_code,
    detect_final_in_text,
    extract_code_blocks,
    format_final_answer,
    resolve_final_var,
)
from .trajectory import (
    TrajectoryEvent,
    TrajectoryEventType,
    TrajectoryLogger,
    TrajectoryViewer,
    compare_trajectories,
    load_trajectory,
)

__all__ = [
    # Environments
    "DSPyCodingRLMEnvironment",
    "EnvironmentActionResult",
    "EnvironmentDoctorCheck",
    "GenericRLMEnvironment",
    "PureRLMConfig",
    "PureRLMEnvironment",
    "RLMRewardProfile",
    # REPL types
    "ImmutableHistory",
    "ImmutableHistoryEntry",
    "REPLEntry",
    "REPLHistory",
    "REPLResult",
    "REPLVariable",
    # Task Signature
    "TaskSignature",
    # Code Interpreter
    "CodeInterpreter",
    "CodeResult",
    "LocalInterpreter",
    "DockerPersistentInterpreter",
    "MockInterpreter",
    # Monty Interpreter (sandboxed Rust-based Python)
    "MontyInterpreter",
    "MontyCodeResult",
    "MontyCodeValidator",
    "MontyExecutionStats",
    "create_rlm_monty_interpreter",
    # Termination
    "FINAL",
    "FINAL_VAR",
    "SUBMIT",
    "FinalDetection",
    "FinalOutput",
    "SubmitOutput",
    "detect_final_in_code",
    "detect_final_in_text",
    "extract_code_blocks",
    "format_final_answer",
    "resolve_final_var",
    # Events (fine-grained)
    "RLMEventBus",
    "RLMEventCollector",
    "RLMEventData",
    "RLMEventType",
    "RLMRuntimeEvent",
    # Memory compaction
    "CompactionConfig",
    "CompactionResult",
    "ConversationMemory",
    "MemoryCompactor",
    # Comparison
    "ComparisonResult",
    "Paradigm",
    "ParadigmComparator",
    "ParadigmResult",
    "create_comparison_report",
    # Frameworks
    "FrameworkAdapterRegistry",
    "FrameworkEpisodeResult",
    "FrameworkStepRecord",
    "DSPyRLMFrameworkAdapter",
    "ADKRLMFrameworkAdapter",
    "GoogleADKFrameworkAdapter",
    "PydanticAIFrameworkAdapter",
    "DeepAgentsFrameworkAdapter",
    # Runner
    "RLMBenchmarkComparison",
    "RLMBenchmarkReport",
    "RLMBenchmarkResult",
    "RLMJudgeResult",
    "RLMRunResult",
    "RLMRunner",
    # Observability
    "RLMObservability",
    "RLMObservabilitySink",
    "LocalJSONLSink",
    "MLflowSink",
    "OpenTelemetrySink",
    "LangSmithSink",
    "LangFuseSink",
    "LogfireSink",
    "CompositeSink",
    "create_otel_sink_from_env",
    "create_langsmith_sink_from_env",
    "create_langfuse_sink_from_env",
    "create_logfire_sink_from_env",
    "create_all_sinks_from_env",
    # Trajectory logging
    "TrajectoryEvent",
    "TrajectoryEventType",
    "TrajectoryLogger",
    "TrajectoryViewer",
    "compare_trajectories",
    "load_trajectory",
    # Configuration
    "RLMConfig",
    "BenchmarkConfig",
    "MCPServerConfig",
    "SandboxConfig",
    "TrajectoryConfig",
    "generate_sample_config",
    "get_default_config",
    # Leaderboard
    "Leaderboard",
    "LeaderboardEntry",
    "LeaderboardFilter",
    "RankingMetric",
    "RankingResult",
    "SortOrder",
    "leaderboard_cli",
    "aggregate_by_field",
    "compute_trend",
    # Session Replay
    "SessionSnapshot",
    "SessionRecorder",
    "SessionReplayer",
    "SessionStore",
    "SessionEvent",
    "SessionEventType",
    "StepState",
    "SessionComparison",
    "compare_sessions",
    "load_session",
    "create_recorder",
    # Policy Lab
    "Policy",
    "PolicyRegistry",
    "RewardPolicy",
    "ActionSelectionPolicy",
    "CompactionPolicy",
    "TerminationPolicy",
    "DefaultRewardPolicy",
    "StrictRewardPolicy",
    "LenientRewardPolicy",
    "ResearchRewardPolicy",
    "GreedyActionPolicy",
    "SamplingActionPolicy",
    "BeamSearchActionPolicy",
    "MCTSActionPolicy",
    "LLMCompactionPolicy",
    "DeterministicCompactionPolicy",
    "SlidingWindowCompactionPolicy",
    "HierarchicalCompactionPolicy",
    "FinalPatternTerminationPolicy",
    "RewardThresholdTerminationPolicy",
    "ConfidenceTerminationPolicy",
    # Approval / HITL
    "ApprovalGate",
    "ApprovalRequest",
    "ApprovalResponse",
    "ApprovalStatus",
    "ApprovalPolicy",
    "RiskAssessor",
    "ToolRiskLevel",
    "RiskAssessment",
    "ApprovalHandler",
    "ConsoleApprovalHandler",
    "AutoApproveHandler",
    "AutoDenyHandler",
    "CallbackApprovalHandler",
    "ApprovalAuditLog",
    "AuditEntry",
]
