import unittest

from utils.clause_parser import split_into_clauses


class ClauseParserTests(unittest.TestCase):
    def test_numeric_clauses(self) -> None:
        text = "1 First clause\n2 Second clause"
        self.assertEqual(split_into_clauses(text), ["1 First clause", "2 Second clause"])

    def test_nested_numeric_clauses(self) -> None:
        text = "1.1 Intro\n1.2 Scope\n2 Main"
        self.assertEqual(split_into_clauses(text), ["1.1 Intro", "1.2 Scope", "2 Main"])

    def test_alpha_and_roman_markers(self) -> None:
        text = "(a) Confidentiality obligations\n(b) Indemnity obligations"
        self.assertEqual(
            split_into_clauses(text),
            ["(a) Confidentiality obligations", "(b) Indemnity obligations"],
        )

        roman = "(i) First condition\n(ii) Second condition"
        self.assertEqual(split_into_clauses(roman), ["(i) First condition", "(ii) Second condition"])

    def test_uppercase_marker(self) -> None:
        text = "A. Payment Terms\nB. Limitation of Liability"
        self.assertEqual(split_into_clauses(text), ["A. Payment Terms", "B. Limitation of Liability"])

    def test_article_and_section(self) -> None:
        text = "Article 1 Purpose\nArticle 2 Scope"
        self.assertEqual(split_into_clauses(text), ["Article 1 Purpose", "Article 2 Scope"])

        section = "Section 1.1 Definitions\nSection 1.2 Interpretation"
        self.assertEqual(
            split_into_clauses(section),
            ["Section 1.1 Definitions", "Section 1.2 Interpretation"],
        )

    def test_article_roman_headings(self) -> None:
        text = "ARTICLE I General Terms\nARTICLE II Covenants"
        self.assertEqual(split_into_clauses(text), ["ARTICLE I General Terms", "ARTICLE II Covenants"])

    def test_mixed_headings(self) -> None:
        text = (
            "ARTICLE I General Terms\n"
            "Section 1.01 Definitions\n"
            "Section 1.02 Interpretation\n"
            "2 Payment"
        )
        self.assertEqual(
            split_into_clauses(text),
            [
                "ARTICLE I General Terms",
                "Section 1.01 Definitions",
                "Section 1.02 Interpretation",
                "2 Payment",
            ],
        )

    def test_paragraph_fallback(self) -> None:
        text = "First paragraph.\n\nSecond paragraph."
        self.assertEqual(split_into_clauses(text), ["First paragraph.", "Second paragraph."])

    def test_custom_pattern(self) -> None:
        text = "Clause-1: Start\nClause-2: End"
        clauses = split_into_clauses(
            text,
            header_patterns=[r"(?m)^\s*Clause-(?P<number>\d+):\s+"],
        )
        self.assertEqual(clauses, ["Clause-1: Start", "Clause-2: End"])

    def test_structured_output(self) -> None:
        text = "1   First clause  \n2 Second    clause"
        structured = split_into_clauses(text, structured=True)
        self.assertEqual(
            structured,
            [
                {"number": "1", "text": "First clause"},
                {"number": "2", "text": "Second clause"},
            ],
        )

    def test_disable_whitespace_collapse(self) -> None:
        text = "1 First clause\ncontinued\n2 Second clause"
        clauses = split_into_clauses(text, collapse_whitespace=False)
        self.assertEqual(clauses, ["1 First clause\ncontinued", "2 Second clause"])


if __name__ == "__main__":
    unittest.main()
