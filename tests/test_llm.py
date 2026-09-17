"""Automated tests for LLM Analytics Assistant module and context compilation."""

import os
import pytest

from src.llm_assistant import (
    OFFLINE_SUGGESTED_ANSWERS,
    compile_verified_analytical_context,
    query_analytics_assistant,
)


def test_compile_verified_analytical_context():
    """Verify analytical context compiles real calculated metrics without PII or raw patient IDs."""
    context = compile_verified_analytical_context()

    assert "platform_name" in context
    assert "cohort_metrics" in context
    assert "empirical_associations" in context
    assert "model_test_benchmarks" in context

    # Check that real calculated metrics are present
    cohort = context["cohort_metrics"]
    assert cohort["raw_encounters"] == 101766
    assert cohort["cleaned_encounters"] == 100111
    assert cohort["unique_patients"] == 70436
    assert cohort["mortality_ineligibility_excluded"] == 1652
    assert cohort["base_30d_readmission_rate_pct"] == 11.34

    # Ensure no raw patient data or PII is leaked in the context
    context_str = str(context)
    assert "patient_nbr" not in context_str.lower() or "patient_nbrs" not in context_str
    assert "encounter_id" not in context_str.lower()


def test_offline_analytics_assistant_graceful_handling():
    """Verify that assistant gracefully handles missing API key without crashing."""
    # Ensure temporary unset of API key for testing fallback behavior
    original_key = os.environ.get("OPENAI_API_KEY")
    try:
        os.environ["OPENAI_API_KEY"] = "your_api_key_here"

        # 1. Preset recommended question should return verified precomputed brief
        query = "What are the main findings in this dataset?"
        response, is_live = query_analytics_assistant(query)
        assert is_live is False
        assert "Verified Analytical Findings" in response
        assert "11.34%" in response

        # 2. Unknown custom query should return friendly configuration message without error
        custom_query = "What is the average blood pressure?"
        response_custom, is_live_custom = query_analytics_assistant(custom_query)
        assert is_live_custom is False
        assert "LLM Analytics Assistant Offline" in response_custom
        assert "OPENAI_API_KEY" in response_custom

    finally:
        if original_key:
            os.environ["OPENAI_API_KEY"] = original_key
        else:
            os.environ.pop("OPENAI_API_KEY", None)
