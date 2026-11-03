"""Paper Eq. 7–9 execution loop; no score-based acceptance or hidden repair.

The policy sees the accumulated knowledge and every previous code/feedback pair.
The current candidate is returned at the budget limit, even if an earlier one
scored better. Benchmark success remains a separate post-hoc measurement.
"""
from __future__ import annotations

from typing import Any

from .paper_execution import PaperExecution, OrchestrationResult
from .generator_agent import GeneratorAgent
from .physics_aware_critic import PhysicsAwareCritic
from .feedback_transformer import ConstraintEvidenceRenderer, FeedbackMessage, RuleBasedFeedbackTransformer
from .pymatgen_api_guard import get_pymatgen_api_guard


class PaperFeedbackParser(RuleBasedFeedbackTransformer):
    """Dependency-ordered repair text, without aggregate critic scores."""

    CATEGORY_PRIORITY = {
        "execution": 0, "geometry": 1, "geometric": 1,
        "identity": 2, "chemistry": 2, "chemical": 2,
        "thermodynamics": 3, "thermodynamic": 3,
        "alignment": 4, "semantic": 4,
    }

    def __init__(self):
        super().__init__(include_metrics=False)

    def transform(self, execution_result, critic_state, task_description, iteration=1):
        execution_ok = bool(execution_result.get("success")) and not execution_result.get("partial_execution_error")
        lines, queries = [], []
        if not execution_ok:
            trace = str(execution_result.get("output", "No structure returned"))
            lines.extend([
                "Fix the execution failure before addressing physical constraints.",
                "Execution evidence: " + self._summarize_execution_error(trace),
                "Return a complete executable script producing a pymatgen Structure named structure.",
            ])
            queries.extend(self._queries_for_execution_error(trace, task_description))
        issues = [] if critic_state is None else sorted(
            critic_state.issues,
            key=lambda issue: (self.CATEGORY_PRIORITY.get(issue.category, 4),
                               {"critical": 0, "major": 1, "minor": 2}.get(issue.severity, 3)),
        )
        for issue in issues:
            lines.extend(ConstraintEvidenceRenderer.issue_lines(issue))
            queries.extend(self._queries_for_issue(issue.category, issue.description))
            queries.extend(self._queries_for_issue_metrics(issue.metrics))
        stop = execution_ok and critic_state is not None and not issues
        if lines:
            lines.extend(ConstraintEvidenceRenderer.invariant_lines(critic_state))
            lines.append("Preserve all task requirements and resolve the earliest dependency first. "
                         "Make the smallest local construction change; do not replace the whole structure "
                         "unless the measured failure requires it. Recheck the listed invariants after editing.")
        return FeedbackMessage(
            text="\n".join(lines), repair_queries=self._dedupe(queries), stop=stop,
            metadata={"iteration": iteration, "execution_success": execution_ok,
                      "parser": "paper_dependency_order_evidence_v2", "issue_count": len(issues)},
        )


class PaperAlignedOrchestrator(PaperExecution):
    """Use reusable execution tools, with an explicit paper-only control loop."""

    def __init__(
        self, llm_client, rag_system=None, max_iterations=3,
        enable_refinement=True, enable_reflection=False, enable_react=False,
        react_max_steps=5, project_root=None, enable_mlip=False,
        mlip_model="chgnet", mlip_device="auto", feedback_transformer="rule",
        feedback_context_expansion=True, critic=None,
    ):
        # Legacy constructor arguments remain accepted for callers; this profile
        # always uses one code policy and deterministic paper feedback.
        self.llm_client = llm_client
        self.rag_system = rag_system
        self.max_iterations = max_iterations
        self.enable_reflection = False
        self.enable_react = False
        self.enable_mlip = enable_mlip
        self.mlip_model = mlip_model
        self.mlip_device = mlip_device
        self.generator = GeneratorAgent(llm_client, rag_system)
        self.method_critic = critic or PhysicsAwareCritic(
            enable_thermo=enable_mlip, mlip_model=mlip_model,
            mlip_device=mlip_device,
        )
        self.feedback_transformer = PaperFeedbackParser()
        self.feedback_context_expansion = feedback_context_expansion

    @staticmethod
    def _append_context(contexts: list[str], context: str) -> None:
        if context and context.strip() and context.strip() not in contexts:
            contexts.append(context.strip())

    def _paper_retrieve(self, query: str, task_input: dict, k: int) -> tuple[list[dict], dict]:
        if self.rag_system is None:
            raise RuntimeError("Paper configuration requires a retrieval backend")
        results = self.rag_system.retrieve(query, k=k, task_input=task_input)
        audit = {
            "query": query,
            "records": [{"id": r.get("id"), "score": r.get("score"),
                         "source": r.get("source"), "retrieval": r.get("retrieval", {})}
                        for r in results],
            "backend_status": getattr(self.rag_system, "last_retrieval_metadata", {}),
        }
        return results, self._json_safe(audit)

    def _merge_knowledge(self, contexts, records, seen_ids):
        for record in records:
            key = str(record.get("id") or record.get("content"))
            if key in seen_ids:
                continue
            seen_ids.add(key)
            # Retrieval returns entry-sized documents. Keep each selected entry
            # once, including its full content, throughout the task.
            context = f"[{record.get('source', 'knowledge')} / {key}]\n{record.get('content', '')}"
            self._append_context(contexts, context)

    def _paper_repair_rules(self, query, task_input):
        if not hasattr(self.rag_system, "retrieve_rules"):
            return [], {}
        records = self.rag_system.retrieve_rules(query, k=3, task_input=task_input)
        return records, self._json_safe({
            "query": query, "mode": "independent_rule_repair_supplement",
            "selected_ids": [r.get("id") for r in records],
            "backend_status": getattr(self.rag_system, "last_retrieval_metadata", {}),
        })

    def run(self, task_description: str, task_input: dict[str, Any] | None = None):
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        task_input = dict(task_input or {})
        task_input.setdefault("task_type", getattr(self.rag_system, "task_type", None))
        contexts, history, evaluations, retrieval_log = [], [], [], []
        seen_knowledge_ids = set()
        # API compatibility is a fixed construction reference, not a hidden repair.
        self._append_context(contexts, get_pymatgen_api_guard(task_description, task_input))
        records, audit = self._paper_retrieve(task_description, task_input, k=9)
        self._merge_knowledge(contexts, records, seen_knowledge_ids)
        retrieval_log.append(audit)
        plan = {"task_type": task_input.get("task_type"), "policy": "paper_code_synthesis"}
        accepted, total_tokens = False, 0
        current_code, current_structure, critic_state = "", None, None
        execution = {"success": False, "output": "No attempt executed"}
        for iteration in range(1, self.max_iterations + 1):
            # No silent history truncation: all previous c_i,f_i enter the policy.
            feedback_history = "\n\n".join(
                f"Round {row['iteration']} program:\n```python\n{row['code']}\n```\n"
                f"Repair feedback:\n{row['feedback']['text']}" for row in history
            )
            response = self.generator.process({
                "plan": plan, "task_description": task_description,
                "task_input": task_input, "rag_context": "\n\n".join(contexts),
                "feedback": feedback_history,
            })
            total_tokens += int(response.metadata.get("tokens_used", 0) or 0)
            current_code = response.content
            execution = self._execute_code(current_code) if response.success else {
                "success": False, "output": response.reasoning, "structure": None,
            }
            # The paper's failed-execution branch has z_t = empty, even when a
            # partially constructed Structure was serialized by the executor.
            if execution.get("partial_execution_error"):
                execution = {**execution, "success": False, "structure": None}
            current_structure = execution.get("structure") if execution.get("success") else None
            critic_state = None
            if current_structure is not None:
                critic_state = self.method_critic.evaluate(
                    structure=current_structure, task_description=task_description,
                    task_input=task_input, deterministic_evaluation=None,
                )
            feedback = self.feedback_transformer.transform(
                execution_result=execution, critic_state=critic_state,
                task_description=task_description, iteration=iteration,
            )
            evaluation = critic_state.to_evaluation() if critic_state else {"execution_failed": True}
            evaluation["feedback"] = feedback.to_dict()
            evaluations.append(self._json_safe(evaluation))
            history.append(self._json_safe({
                "iteration": iteration, "code": current_code,
                "execution": {"success": bool(execution.get("success")),
                              "output": execution.get("output", "")},
                "critic_state": critic_state.to_dict() if critic_state else None,
                "feedback": feedback.to_dict(),
            }))
            if feedback.stop:
                accepted = True
                break
            if iteration < self.max_iterations and self.feedback_context_expansion:
                for query in feedback.repair_queries:
                    records, audit = self._paper_retrieve(query, task_input, k=3)
                    self._merge_knowledge(contexts, records, seen_knowledge_ids)
                    retrieval_log.append(audit)
                    records, audit = self._paper_repair_rules(query, task_input)
                    self._merge_knowledge(contexts, records, seen_knowledge_ids)
                    if records:
                        retrieval_log.append(audit)
        return OrchestrationResult(
            success=current_structure is not None, final_code=current_code,
            structure=current_structure, iterations=len(history), total_tokens=total_tokens,
            plan=plan, evaluations=evaluations,
            metadata=self._json_safe({
                "method_profile": "paper_aligned_v1", "execution_result": execution,
                "final_evaluation": evaluations[-1], "state_history": history,
                "active_context_count": len(contexts), "retrieval_log": retrieval_log,
                "feedback_transformer": "PaperFeedbackParser",
                "accepted_by_methods_critic": accepted,
                "acceptance_decision": {"accept": accepted,
                    "reason": "no_execution_or_critic_issues" if accepted else "iteration_budget"},
                "selected_best_candidate": False, "selected_from_best_so_far": False,
                "selected_iteration": len(history), "automatic_structure_repair": False,
                "full_history_conditioning": True,
            }),
        )
