from typing import Any, Optional
from llm import LocalHuggingFaceChatModel


class HyDE:
    """Simple HyDE-style query transformer.

    This class generates a hypothetical document for a given user query
    using an LLM, and can optionally return a combined query that
    includes the original user query plus the generated hypothetical
    document. It is intentionally lightweight so you can wire it into
    your retrieval pipeline or into `TransformQueryEngine`-style code.

    Args:
        llm: An object exposing an `invoke(input)` method. If not
            provided, the project's `LocalHuggingFaceChatModel` is used.
        prompt_template: A template string that will receive the query
            as `{query}`. The LLM output is treated as the hypothetical
            document.
        include_original: If True, `transform()` returns the original
            query concatenated with the hypothetical document.
    """

    DEFAULT_PROMPT = (
        "Given the user question below, write a short hypothetical document "
        "that contains plausible facts and context that would help answer the question.\n\n"
        "Constraints:\n"
        "- Write ONLY 3 to 5 complete sentences.\n"
        "- Do NOT add explanations, lists, or extra commentary.\n"
        "- Do NOT mention that this is hypothetical.\n\n"
        "Question: {query}\n\n"
        "Hypothetical document:"
    )

    def __init__(
        self,
        llm: Optional[Any] = None,
        prompt_template: str = DEFAULT_PROMPT,
        include_original: bool = True,
        max_tokens: int = 256,
    ) -> None:
        self.llm = llm or LocalHuggingFaceChatModel()
        self.prompt_template = prompt_template
        self.include_original = include_original
        self.max_tokens = max_tokens

    def _build_prompt(self, query: str) -> str:
        return self.prompt_template.format(query=query)

    def generate_hypothetical(self, query: str) -> str:
        """Return a single hypothetical document string for `query`."""
        prompt = self._build_prompt(query)
        # Use the LLM's invoke method. We accept either raw string input
        # or objects with `.invoke()`.
        if hasattr(self.llm, "invoke"):
            out = self.llm.invoke(prompt)
            # If the LLM wrapper returns an AIMessage-like object with
            # `.content`, extract it. Otherwise coerce to str.
            if hasattr(out, "content"):
                return str(out.content).strip()
            return str(out).strip()

        # Fallback: call the LLM as a function
        return str(self.llm(prompt)).strip()

    def transform(self, query: str) -> str:
        """Return a transformed query that includes the HyDE document.

        If `include_original` is True the return value will be the original
        query followed by the generated hypothetical document. Otherwise
        only the hypothetical document is returned.
        """
        hyde_doc = self.generate_hypothetical(query)
        if self.include_original:
            return f"{query}\n\n[HyDE document]\n{hyde_doc}"
        return hyde_doc

    __call__ = transform


def main():
    # Quick demo showing how to generate a hypothetical document.
    hyde = HyDE()
    sample_query = "Compare the families of Emma Stone and Ryan Gosling"
    print("Generating hypothetical document for sample query...\n")
    print(hyde.generate_hypothetical(sample_query))


if __name__ == "__main__":
    main()
