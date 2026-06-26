import unittest

from api.core.utils import infer_category, tokenize_ko


class SearchTextProcessingTests(unittest.TestCase):
    def test_compound_query_matches_longer_notice_term(self):
        query_tokens = set(tokenize_ko("해외봉사 어떤 거 있어?"))
        doc_tokens = set(tokenize_ko("해외봉사활동 50기 WFK 청년봉사단 단원 모집 안내"))

        self.assertIn("해외봉사", query_tokens)
        self.assertIn("해외봉사", doc_tokens)
        self.assertIn("해외", doc_tokens)
        self.assertIn("봉사", doc_tokens)

    def test_overseas_volunteer_notice_prefers_volunteer_category(self):
        title = "한성공지[대학사회봉사협의회] [해외봉사활동] 50기 WFK 청년봉사단 단원 모집 안내"
        body = "총 4주 해외파견 혼합형 봉사 프로그램입니다."

        self.assertEqual(infer_category(title, body), "봉사/서포터즈")


if __name__ == "__main__":
    unittest.main()
