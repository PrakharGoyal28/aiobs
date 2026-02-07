import math
from aiobs.evals import SemanticSimilarityEval, EvalInput


class FakeLLM:
    """Fake LLM that returns deterministic embeddings."""

    def embed(self, text: str):
        # Very simple embedding: count character frequencies
        vec = [0.0] * 26
        for c in text.lower():
            if "a" <= c <= "z":
                vec[ord(c) - ord("a")] += 1.0
        return vec


def cosine(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2)


def test_semantic_similarity_pass():
    evaluator = SemanticSimilarityEval()
    llm = FakeLLM()

    inp = EvalInput(
        user_input="Capital of France?",
        model_output="Paris is the capital city of France.",
        expected_output="The capital of France is Paris."
    )

    result = evaluator.evaluate(inp, llm=llm)

    assert result.passed
    assert result.score > evaluator.config.threshold


def test_semantic_similarity_fail():
    evaluator = SemanticSimilarityEval()
    llm = FakeLLM()

    inp = EvalInput(
        user_input="Capital of France?",
        model_output="Berlin is the capital of Germany.",
        expected_output="The capital of France is Paris."
    )

    result = evaluator.evaluate(inp, llm=llm)

    assert not result.passed
    assert result.score < evaluator.config.threshold


def test_semantic_similarity_missing_expected():
    evaluator = SemanticSimilarityEval()
    llm = FakeLLM()

    inp = EvalInput(
        user_input="Capital of France?",
        model_output="Paris is the capital city of France."
    )

    result = evaluator.evaluate(inp, llm=llm)

    assert result.status.name == "ERROR"
