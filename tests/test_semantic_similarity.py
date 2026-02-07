import os
import pytest

from openai import OpenAI
from aiobs.llm import LLM
from aiobs.evals import SemanticSimilarityEval, EvalInput


@pytest.fixture(scope="module")
def llm():
    # Requires OPENAI_API_KEY in environment
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return LLM.from_client(client, model="gpt-4o-mini")


@pytest.fixture
def evaluator():
    return SemanticSimilarityEval()


def test_semantic_similarity_pass(llm, evaluator):
    """Similar meaning should pass the threshold."""
    inp = EvalInput(
        user_input="Capital of France?",
        model_output="Paris is the capital city of France.",
        expected_output="The capital of France is Paris."
    )

    result = evaluator.evaluate(inp, llm=llm)

    assert result.passed is True
    assert result.score > evaluator.config.threshold


def test_semantic_similarity_fail(llm, evaluator):
    """Different meaning should fail."""
    inp = EvalInput(
        user_input="Capital of France?",
        model_output="Berlin is the capital of Germany.",
        expected_output="The capital of France is Paris."
    )

    result = evaluator.evaluate(inp, llm=llm)

    assert result.passed is False
    assert result.score < evaluator.config.threshold


def test_semantic_similarity_missing_expected(llm, evaluator):
    """Should error when expected_output is missing."""
    inp = EvalInput(
        user_input="Capital of France?",
        model_output="Paris is the capital city of France.",
    )

    result = evaluator.evaluate(inp, llm=llm)

    assert result.status.name == "ERROR"
