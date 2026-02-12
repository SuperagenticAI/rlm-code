"""
RLM runtime for RLM Code CLI.

This module provides the core RLM (Recursive Language Model) runtime,
implementing the paradigm from "Recursive Language Models" (Zhang, Kraska, Khattab, 2025).

Key components:
- PureRLMEnvironment: Paper-compliant RLM with context-as-variable
- REPLVariable/REPLHistory: Structured REPL state management
- FINAL/FINAL_VAR: Clean termination patterns
- RLMRunner: Orchestrates RLM execution loops
- ParadigmComparator: Side-by-side comparison of RLM paradigms
- MemoryCompactor: Context window management via summarization
"""

from .comparison import (
    ComparisonResult,
    Paradigm,
    ParadigmComparator,
    ParadigmResult,
    create_comparison_report,
)
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
    FrameworkAdapterRegistry,
    FrameworkEpisodeResult,
    FrameworkStepRecord,
    GoogleADKFrameworkAdapter,
    PydanticAIFrameworkAdapter,
)
from .memory_compaction import (
    CompactionConfig,
    CompactionResult,
    ConversationMemory,
    MemoryCompactor,
)
from .observability import (
    RLMObservability,
    RLMObservabilitySink,
    LocalJSONLSink,
    MLflowSink,
    OpenTelemetrySink,
    LangSmithSink,
    LangFuseSink,
    LogfireSink,
    CompositeSink,
    create_otel_sink_from_env,
    create_langsmith_sink_from_env,
    create_langfuse_sink_from_env,
    create_logfire_sink_from_env,
    create_all_sinks_from_env,
)
from .code_interpreter import CodeInterpreter, CodeResult, LocalInterpreter
from .mock_interpreter import MockInterpreter
from .monty_interpreter import (
    MontyCodeResult,
    MontyCodeValidator,
    MontyExecutionStats,
    MontyInterpreter,
    create_rlm_monty_interpreter,
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
    RLMBenchmarkResult,
    RLMRunner,
    RLMRunResult,
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
from .config_schema import (
    RLMConfig,
    BenchmarkConfig,
    MCPServerConfig,
    SandboxConfig,
    TrajectoryConfig,
    generate_sample_config,
    get_default_config,
)
from .leaderboard import (
    Leaderboard,
    LeaderboardEntry,
    LeaderboardFilter,
    RankingMetric,
    RankingResult,
    SortOrder,
    leaderboard_cli,
    aggregate_by_field,
    compute_trend,
)
from .session_replay import (
    SessionSnapshot,
    SessionRecorder,
    SessionReplayer,
    SessionStore,
    SessionEvent,
    SessionEventType,
    StepState,
    SessionComparison,
    compare_sessions,
    load_session,
    create_recorder,
)
# Policy Lab - Hot-swappable policies
from .policies import (
    Policy,
    PolicyRegistry,
    RewardPolicy,
    ActionSelectionPolicy,
    CompactionPolicy,
    TerminationPolicy,
    # Reward policies
    DefaultRewardPolicy,
    StrictRewardPolicy,
    LenientRewardPolicy,
    ResearchRewardPolicy,
    # Action policies
    GreedyActionPolicy,
    SamplingActionPolicy,
    BeamSearchActionPolicy,
    MCTSActionPolicy,
    # Compaction policies
    LLMCompactionPolicy,
    DeterministicCompactionPolicy,
    SlidingWindowCompactionPolicy,
    HierarchicalCompactionPolicy,
    # Termination policies
    FinalPatternTerminationPolicy,
    RewardThresholdTerminationPolicy,
    ConfidenceTerminationPolicy,
)
# Approval / HITL Gates
from .approval import (
    ApprovalGate,
    ApprovalRequest,
    ApprovalResponse,
    ApprovalStatus,
    ApprovalPolicy,
    RiskAssessor,
    ToolRiskLevel,
    RiskAssessment,
    ApprovalHandler,
    ConsoleApprovalHandler,
    AutoApproveHandler,
    AutoDenyHandler,
    CallbackApprovalHandler,
    ApprovalAuditLog,
    AuditEntry,
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
    "GoogleADKFrameworkAdapter",
    "PydanticAIFrameworkAdapter",
    # Runner
    "RLMBenchmarkComparison",
    "RLMBenchmarkResult",
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
