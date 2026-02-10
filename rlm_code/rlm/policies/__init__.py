"""
Policy Lab - Hot-swappable policies for RLM execution.

Provides pluggable policies for:
- Reward calculation
- Action selection
- Memory compaction
- Termination detection

Usage:
    from rlm_code.rlm.policies import (
        PolicyRegistry,
        RewardPolicy,
        ActionSelectionPolicy,
        CompactionPolicy,
    )

    # Register custom policy
    @PolicyRegistry.register_reward("my_reward")
    class MyRewardPolicy(RewardPolicy):
        def calculate(self, action, result, context):
            return custom_reward_logic(...)

    # Use in runner
    runner = RLMRunner(
        reward_policy="my_reward",
        action_policy="greedy",
        compaction_policy="llm_summary",
    )
"""

from .base import (
    Policy,
    RewardPolicy,
    ActionSelectionPolicy,
    CompactionPolicy,
    TerminationPolicy,
)
from .registry import PolicyRegistry
from .reward_policies import (
    DefaultRewardPolicy,
    StrictRewardPolicy,
    LenientRewardPolicy,
    ResearchRewardPolicy,
)
from .action_policies import (
    GreedyActionPolicy,
    SamplingActionPolicy,
    BeamSearchActionPolicy,
    MCTSActionPolicy,
)
from .compaction_policies import (
    LLMCompactionPolicy,
    DeterministicCompactionPolicy,
    SlidingWindowCompactionPolicy,
    HierarchicalCompactionPolicy,
)
from .termination_policies import (
    FinalPatternTerminationPolicy,
    RewardThresholdTerminationPolicy,
    ConfidenceTerminationPolicy,
)

__all__ = [
    # Base classes
    "Policy",
    "RewardPolicy",
    "ActionSelectionPolicy",
    "CompactionPolicy",
    "TerminationPolicy",
    # Registry
    "PolicyRegistry",
    # Reward policies
    "DefaultRewardPolicy",
    "StrictRewardPolicy",
    "LenientRewardPolicy",
    "ResearchRewardPolicy",
    # Action policies
    "GreedyActionPolicy",
    "SamplingActionPolicy",
    "BeamSearchActionPolicy",
    "MCTSActionPolicy",
    # Compaction policies
    "LLMCompactionPolicy",
    "DeterministicCompactionPolicy",
    "SlidingWindowCompactionPolicy",
    "HierarchicalCompactionPolicy",
    # Termination policies
    "FinalPatternTerminationPolicy",
    "RewardThresholdTerminationPolicy",
    "ConfidenceTerminationPolicy",
]
