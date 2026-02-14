"""
P0 Feature Tests: Policy Lab and Tool Approval / HITL Gates.

Tests for:
- Hot-swappable reward policies
- Hot-swappable action selection policies
- Hot-swappable compaction policies
- Hot-swappable termination policies
- Tool approval and HITL gates
- Risk assessment
- Audit logging
"""

import tempfile
from pathlib import Path

import pytest

from rlm_code.rlm.approval import (
    ApprovalAuditLog,
    ApprovalGate,
    ApprovalPolicy,
    ApprovalStatus,
    AutoApproveHandler,
    AutoDenyHandler,
    RiskAssessor,
    ToolRiskLevel,
)
from rlm_code.rlm.policies import (
    BeamSearchActionPolicy,
    ConfidenceTerminationPolicy,
    DefaultRewardPolicy,
    # Compaction policies
    DeterministicCompactionPolicy,
    # Termination policies
    FinalPatternTerminationPolicy,
    # Action policies
    GreedyActionPolicy,
    HierarchicalCompactionPolicy,
    LenientRewardPolicy,
    MCTSActionPolicy,
    PolicyRegistry,
    ResearchRewardPolicy,
    RewardThresholdTerminationPolicy,
    SamplingActionPolicy,
    SlidingWindowCompactionPolicy,
    StrictRewardPolicy,
)
from rlm_code.rlm.policies.base import (
    ActionResult,
    PolicyContext,
)


class TestPolicyRegistry:
    """Tests for policy registry and hot-swapping."""

    def test_registry_has_default_policies(self):
        """Test that default policies are registered."""
        # Reward policies
        assert "default" in [p["name"] for p in PolicyRegistry.list_reward_policies()]
        assert "strict" in [p["name"] for p in PolicyRegistry.list_reward_policies()]

        # Action policies
        assert "greedy" in [p["name"] for p in PolicyRegistry.list_action_policies()]
        assert "sampling" in [p["name"] for p in PolicyRegistry.list_action_policies()]

    def test_get_reward_policy(self):
        """Test getting reward policy by name."""
        policy = PolicyRegistry.get_reward("default")
        assert isinstance(policy, DefaultRewardPolicy)

        policy = PolicyRegistry.get_reward("strict")
        assert isinstance(policy, StrictRewardPolicy)

    def test_get_action_policy(self):
        """Test getting action policy by name."""
        policy = PolicyRegistry.get_action("greedy")
        assert isinstance(policy, GreedyActionPolicy)

        policy = PolicyRegistry.get_action("sampling")
        assert isinstance(policy, SamplingActionPolicy)

    def test_get_compaction_policy(self):
        """Test getting compaction policy by name."""
        policy = PolicyRegistry.get_compaction("sliding_window")
        assert isinstance(policy, SlidingWindowCompactionPolicy)

    def test_get_termination_policy(self):
        """Test getting termination policy by name."""
        policy = PolicyRegistry.get_termination("final_pattern")
        assert isinstance(policy, FinalPatternTerminationPolicy)

    def test_invalid_policy_raises(self):
        """Test that invalid policy name raises error."""
        with pytest.raises(ValueError, match="Unknown reward policy"):
            PolicyRegistry.get_reward("nonexistent")

    def test_list_all_policies(self):
        """Test listing all policies."""
        all_policies = PolicyRegistry.list_all()

        assert "reward" in all_policies
        assert "action" in all_policies
        assert "compaction" in all_policies
        assert "termination" in all_policies

    def test_create_from_config(self):
        """Test creating policies from config dict."""
        config = {
            "reward": {"name": "strict", "config": {}},
            "action": {"name": "greedy", "config": {}},
        }

        policies = PolicyRegistry.create_from_config(config)

        assert "reward" in policies
        assert isinstance(policies["reward"], StrictRewardPolicy)
        assert "action" in policies
        assert isinstance(policies["action"], GreedyActionPolicy)


class TestRewardPolicies:
    """Tests for reward policy implementations."""

    def test_default_reward_success(self):
        """Test default reward for successful action."""
        policy = DefaultRewardPolicy()

        action = {"action": "run_python", "code": "print('hello')"}
        result = ActionResult(
            action_type="run_python",
            success=True,
            output="hello",
        )
        context = PolicyContext(task="test", step=1)

        signal = policy.calculate(action, result, context)

        assert signal.value > 0
        assert "success" in signal.components
        assert signal.components["success"] > 0

    def test_default_reward_failure(self):
        """Test default reward for failed action."""
        policy = DefaultRewardPolicy()

        action = {"action": "run_python", "code": "raise Error"}
        result = ActionResult(
            action_type="run_python",
            success=False,
            error="Error occurred",
        )
        context = PolicyContext(task="test", step=1)

        signal = policy.calculate(action, result, context)

        assert "failure" in signal.components
        assert signal.components["failure"] < 0

    def test_strict_reward_heavier_penalties(self):
        """Test strict reward has heavier penalties than default."""
        default_policy = DefaultRewardPolicy()
        strict_policy = StrictRewardPolicy()

        action = {"action": "run_python", "code": "x"}
        result = ActionResult(
            action_type="run_python",
            success=False,
            error="NameError",
        )
        context = PolicyContext(task="test", step=1)

        default_signal = default_policy.calculate(action, result, context)
        strict_signal = strict_policy.calculate(action, result, context)

        # Strict should have lower (more negative) reward for failure
        assert strict_signal.value <= default_signal.value

    def test_lenient_reward_more_forgiving(self):
        """Test lenient reward is more forgiving."""
        default_policy = DefaultRewardPolicy()
        lenient_policy = LenientRewardPolicy()

        action = {"action": "run_python", "code": "x"}
        result = ActionResult(
            action_type="run_python",
            success=False,
            error="Error",
        )
        context = PolicyContext(task="test", step=1)

        default_signal = default_policy.calculate(action, result, context)
        lenient_signal = lenient_policy.calculate(action, result, context)

        # Lenient should have higher reward for failure (less penalty)
        assert lenient_signal.value >= default_signal.value

    def test_research_reward_detailed_components(self):
        """Test research reward provides detailed breakdown."""
        policy = ResearchRewardPolicy()

        action = {"action": "run_python", "code": "print('hello world')"}
        result = ActionResult(
            action_type="run_python",
            success=True,
            output="hello world",
            duration_ms=500.0,
        )
        context = PolicyContext(task="test", step=1, max_steps=10)

        signal = policy.calculate(action, result, context)

        # Research policy should have many components
        assert len(signal.components) >= 3
        assert "components" in signal.explanation


class TestActionPolicies:
    """Tests for action selection policies."""

    def test_greedy_selects_highest_score(self):
        """Test greedy policy selects highest scored action."""
        policy = GreedyActionPolicy()

        candidates = [
            {"action": "a", "confidence": 0.3},
            {"action": "b", "confidence": 0.9},
            {"action": "c", "confidence": 0.5},
        ]
        context = PolicyContext(task="test", step=1)

        selected = policy.select(candidates, context)

        assert selected["action"] == "b"

    def test_sampling_respects_temperature(self):
        """Test sampling policy respects temperature."""
        # Low temperature should be more deterministic
        policy = SamplingActionPolicy(config={"temperature": 0.1})

        candidates = [
            {"action": "high", "confidence": 0.9},
            {"action": "low", "confidence": 0.1},
        ]
        context = PolicyContext(task="test", step=1)

        # Run multiple times - low temp should mostly pick high confidence
        selections = [policy.select(candidates, context)["action"] for _ in range(20)]
        high_count = sum(1 for s in selections if s == "high")

        assert high_count >= 15  # Should be mostly "high"

    def test_beam_search_rank(self):
        """Test beam search ranking."""
        policy = BeamSearchActionPolicy(config={"beam_width": 3})

        candidates = [
            {"action": "a", "confidence": 0.5},
            {"action": "b", "confidence": 0.8},
            {"action": "c", "confidence": 0.3},
        ]
        context = PolicyContext(task="test", step=2)

        ranked = policy.rank(candidates, context)

        # Should be sorted by adjusted score
        assert len(ranked) == 3
        assert ranked[0][0]["action"] == "b"  # Highest confidence first

    def test_mcts_exploration(self):
        """Test MCTS explores unexplored actions."""
        policy = MCTSActionPolicy()

        candidates = [
            {"action": "explored", "confidence": 0.5},
            {"action": "unexplored", "confidence": 0.3},
        ]
        context = PolicyContext(task="test", step=1)

        # Simulate that "explored" has been visited
        policy._visits["explored:0"] = 10
        policy._values["explored:0"] = 5.0

        selected = policy.select(candidates, context)

        # Should prefer unexplored due to UCB exploration bonus
        assert selected["action"] == "unexplored"


class TestCompactionPolicies:
    """Tests for compaction policies."""

    def test_sliding_window_keeps_recent(self):
        """Test sliding window keeps only recent entries."""
        policy = SlidingWindowCompactionPolicy(config={"window_size": 3})

        history = [{"step": i, "action": f"action_{i}"} for i in range(10)]
        context = PolicyContext(task="test", step=10, history=history)

        assert policy.should_compact(context) is True

        compacted, summary = policy.compact(history, context)

        # Should have summary + 3 recent
        assert len(compacted) == 4
        assert compacted[-1]["step"] == 9  # Most recent

    def test_deterministic_compaction(self):
        """Test deterministic compaction."""
        policy = DeterministicCompactionPolicy(config={"max_entries": 5})

        history = [{"step": i, "action": "run_python", "output": f"output {i}"} for i in range(10)]
        context = PolicyContext(task="test", step=10, history=history)

        assert policy.should_compact(context) is True

        compacted, summary = policy.compact(history, context)

        assert len(compacted) < len(history)
        assert "run_python" in summary  # Action count in summary

    def test_hierarchical_compaction(self):
        """Test hierarchical multi-level compaction."""
        policy = HierarchicalCompactionPolicy(
            config={
                "recent_window": 2,
                "medium_window": 3,
                "compress_threshold": 6,
            }
        )

        history = [{"step": i, "action": "run_python"} for i in range(10)]
        context = PolicyContext(task="test", step=10, history=history)

        assert policy.should_compact(context) is True

        compacted, summary = policy.compact(history, context)

        # Should have tiered structure
        assert any(e.get("tier") == "old" for e in compacted if isinstance(e, dict))


class TestTerminationPolicies:
    """Tests for termination policies."""

    def test_final_pattern_detects_final(self):
        """Test final pattern detection."""
        policy = FinalPatternTerminationPolicy()

        result = ActionResult(
            action_type="run_python",
            success=True,
            output="FINAL('The answer is 42')",
        )
        context = PolicyContext(task="test", step=1)

        should_term, answer = policy.should_terminate(result, context)

        assert should_term is True
        assert answer == "The answer is 42"

    def test_final_pattern_no_match(self):
        """Test final pattern with no match."""
        policy = FinalPatternTerminationPolicy()

        result = ActionResult(
            action_type="run_python",
            success=True,
            output="Processing... result=42",
        )
        context = PolicyContext(task="test", step=1)

        should_term, answer = policy.should_terminate(result, context)

        assert should_term is False
        assert answer is None

    def test_reward_threshold_termination(self):
        """Test reward threshold termination."""
        policy = RewardThresholdTerminationPolicy(
            config={
                "min_reward_threshold": 0.5,
            }
        )

        result = ActionResult(action_type="run_python", success=True)
        context = PolicyContext(
            task="test",
            step=1,
            metrics={"last_reward": 0.6},
        )

        should_term, _ = policy.should_terminate(result, context)

        assert should_term is True

    def test_confidence_termination(self):
        """Test confidence-based termination."""
        policy = ConfidenceTerminationPolicy(
            config={
                "confidence_threshold": 0.9,
                "min_steps_before_termination": 1,
            }
        )

        result = ActionResult(
            action_type="run_python",
            success=True,
            metadata={"confidence": 0.95},
        )
        context = PolicyContext(task="test", step=2)

        should_term, _ = policy.should_terminate(result, context)

        assert should_term is True


class TestRiskAssessment:
    """Tests for risk assessment."""

    def test_safe_action(self):
        """Test safe action assessment."""
        assessor = RiskAssessor()

        action = {"action": "run_python", "code": "print('hello')"}
        assessment = assessor.assess(action)

        assert assessment.level == ToolRiskLevel.SAFE

    def test_high_risk_file_delete(self):
        """Test high risk for file deletion."""
        assessor = RiskAssessor()

        action = {"action": "run_python", "code": "os.remove('/path/to/file')"}
        assessment = assessor.assess(action)

        assert assessment.level == ToolRiskLevel.HIGH
        assert not assessment.reversible

    def test_critical_risk_rm_rf(self):
        """Test critical risk for rm -rf."""
        assessor = RiskAssessor()

        # Test bash-style rm -rf command
        action = {"action": "run_bash", "code": "rm -rf /tmp/test"}
        assessment = assessor.assess(action)

        assert assessment.level == ToolRiskLevel.CRITICAL

    def test_medium_risk_file_write(self):
        """Test medium risk for file write."""
        assessor = RiskAssessor()

        action = {"action": "run_python", "code": "open('file.txt', 'w').write('data')"}
        assessment = assessor.assess(action)

        assert assessment.level in (ToolRiskLevel.MEDIUM, ToolRiskLevel.LOW)

    def test_affected_resources_extraction(self):
        """Test extraction of affected resources."""
        assessor = RiskAssessor()

        action = {"action": "run_python", "code": "Path('/home/user/data.txt').unlink()"}
        assessment = assessor.assess(action)

        assert any("file:" in r for r in assessment.affected_resources)


class TestApprovalGate:
    """Tests for approval gate."""

    def test_check_safe_action(self):
        """Test checking safe action."""
        gate = ApprovalGate(policy=ApprovalPolicy.CONFIRM_HIGH_RISK)

        action = {"action": "run_python", "code": "print('hello')"}
        request = gate.check_action(action)

        assert request.requires_approval is False

    def test_check_high_risk_action(self):
        """Test checking high risk action."""
        gate = ApprovalGate(policy=ApprovalPolicy.CONFIRM_HIGH_RISK)

        action = {"action": "run_python", "code": "os.remove('/important/file')"}
        request = gate.check_action(action)

        assert request.requires_approval is True

    @pytest.mark.asyncio
    async def test_auto_approve_handler(self):
        """Test auto-approve handler."""
        handler = AutoApproveHandler()
        gate = ApprovalGate(
            policy=ApprovalPolicy.CONFIRM_HIGH_RISK,
            approval_handler=handler.handle,
        )

        action = {"action": "run_python", "code": "os.remove('/file')"}
        request = gate.check_action(action)

        response = await gate.request_approval(request)

        assert response.approved is True
        assert response.status == ApprovalStatus.AUTO_APPROVED

    @pytest.mark.asyncio
    async def test_auto_deny_handler(self):
        """Test auto-deny handler."""
        handler = AutoDenyHandler()
        gate = ApprovalGate(
            policy=ApprovalPolicy.CONFIRM_HIGH_RISK,
            approval_handler=handler.handle,
        )

        action = {"action": "run_python", "code": "os.remove('/file')"}
        request = gate.check_action(action)

        response = await gate.request_approval(request)

        assert response.approved is False
        assert response.status == ApprovalStatus.AUTO_DENIED

    def test_manual_approve(self):
        """Test manual approval."""
        gate = ApprovalGate(policy=ApprovalPolicy.CONFIRM_HIGH_RISK)

        action = {"action": "run_python", "code": "rm -rf /"}
        request = gate.check_action(action)

        response = gate.approve(request.request_id, reason="Verified safe")

        assert response.approved is True
        assert response.status == ApprovalStatus.APPROVED

    def test_manual_deny(self):
        """Test manual denial."""
        gate = ApprovalGate(policy=ApprovalPolicy.CONFIRM_HIGH_RISK)

        action = {"action": "run_python", "code": "rm -rf /"}
        request = gate.check_action(action)

        response = gate.deny(request.request_id, reason="Too risky")

        assert response.approved is False
        assert response.status == ApprovalStatus.DENIED


class TestAuditLog:
    """Tests for approval audit logging."""

    def test_log_entry_creation(self):
        """Test creating audit log entries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "audit.jsonl"
            audit = ApprovalAuditLog(log_file=log_file)

            # Use CONFIRM_ALL policy so even safe actions require approval
            gate = ApprovalGate(audit_log=audit, policy=ApprovalPolicy.CONFIRM_ALL)

            action = {"action": "run_python", "code": "print('test')"}
            request = gate.check_action(action)
            response = gate.approve(request.request_id)

            entries = audit.get_entries()
            assert len(entries) == 1
            assert entries[0].approved is True

    def test_audit_summary(self):
        """Test audit summary statistics."""
        audit = ApprovalAuditLog()
        # Use CONFIRM_ALL policy so all actions require approval and get logged
        gate = ApprovalGate(audit_log=audit, policy=ApprovalPolicy.CONFIRM_ALL)

        # Create some entries
        for i in range(5):
            action = {"action": "run_python", "code": f"action_{i}"}
            request = gate.check_action(action)
            if i % 2 == 0:
                gate.approve(request.request_id)
            else:
                gate.deny(request.request_id)

        summary = audit.get_summary()

        assert summary["total"] == 5
        assert summary["approved"] == 3
        assert summary["denied"] == 2

    def test_audit_export_report(self):
        """Test exporting audit report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audit = ApprovalAuditLog()
            gate = ApprovalGate(audit_log=audit)

            action = {"action": "run_python", "code": "test"}
            request = gate.check_action(action)
            gate.approve(request.request_id)

            report_path = Path(tmpdir) / "report.md"
            audit.export_report(report_path)

            assert report_path.exists()
            content = report_path.read_text()
            assert "Approval Audit Report" in content


class TestPolicyIntegration:
    """Integration tests for policies working together."""

    def test_reward_with_context(self):
        """Test reward calculation with full context."""
        policy = ResearchRewardPolicy()

        action = {"action": "final", "code": "FINAL('answer')"}
        result = ActionResult(
            action_type="final",
            success=True,
            output="answer",
            duration_ms=100.0,
        )
        context = PolicyContext(
            task="Find the answer",
            step=2,
            max_steps=10,
            history=[{"step": 1}],
            variables={"result": "answer"},
        )

        signal = policy.calculate(action, result, context)

        assert signal.value > 0
        assert "final_success" in signal.components

    def test_termination_with_variables(self):
        """Test termination with FINAL_VAR."""
        policy = FinalPatternTerminationPolicy()

        result = ActionResult(
            action_type="run_python",
            success=True,
            output="FINAL_VAR('my_result')",
        )
        context = PolicyContext(
            task="test",
            step=1,
            variables={"my_result": "The computed answer"},
        )

        should_term, answer = policy.should_terminate(result, context)

        assert should_term is True
        assert answer == "The computed answer"
