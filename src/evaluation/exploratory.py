"""Supplementary observable-constraint success; not the paper's strict TSR."""


def summarize_relaxed_tsr(rows):
    """Ignore unknown task constraints, but retain known failures and all attempts.

    Require confirmed structural validity and at least one observed hard task
    constraint. All-unknown evidence is not treated as successful completion.
    """
    total = passed = unknown_samples = observed = required = 0
    for row in rows:
        total += 1
        protocol = row.get("metadata", {}).get("evaluation_protocol", {})
        hard = [c for c in protocol.get("task_constraints", []) if c.get("hard", True)]
        known = [c for c in hard if isinstance(c.get("satisfied"), bool) and c.get("status") != "unknown"]
        required += len(hard)
        observed += len(known)
        unknown_samples += int(len(known) != len(hard))
        passed += int(protocol.get("structural_valid") is True and bool(known)
                      and all(c["satisfied"] for c in known))
    return {"relaxed_task_satisfied": passed, "relaxed_tsr": passed / total if total else 0.0,
            "relaxed_tsr_definition": "confirmed structural validity; all observed hard constraints pass; at least one observed; unknown excluded",
            "relaxed_samples_with_unknown_constraints": unknown_samples,
            "relaxed_observed_hard_constraints": observed, "relaxed_required_hard_constraints": required,
            "relaxed_is_paper_tsr": False}
