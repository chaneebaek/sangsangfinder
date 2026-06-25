import unittest
from unittest import mock

from api.services import search_service


class QueryRoutingToggleTests(unittest.TestCase):
    def test_personalized_query_expands_route_with_profile_terms(self):
        route = {
            "original_query": "내 학과에 맞는 창업 공모전 찾아줘",
            "search_query": "내 학과에 맞는 창업 공모전 찾아줘 창업 공모전",
            "intent": "기타",
            "confidence": 0.7,
            "method": "heuristic",
            "features": {
                "must_keep_terms": ["공모전"],
                "expanded_terms": [],
                "entities": ["공모전"],
                "time_terms": [],
                "section_targets": [],
            },
        }
        profile = {
            "college": "IT공과대학",
            "track": "컴퓨터공학부",
            "grade": "3학년",
            "interests": ["창업", "AI"],
            "region": "서울",
        }

        personalized = search_service._apply_personalization_to_route(
            route,
            "내 학과에 맞는 창업 공모전 찾아줘",
            profile,
        )

        self.assertEqual(personalized["intent"], "conditional")
        self.assertGreaterEqual(personalized["confidence"], 0.88)
        self.assertTrue(personalized["personalization"]["applied"])
        self.assertIn("컴퓨터공학부", personalized["search_query"])
        self.assertIn("3학년", personalized["search_query"])
        self.assertIn("창업", personalized["features"]["expanded_terms"])
        self.assertNotIn("AI", personalized["features"]["expanded_terms"])
        self.assertNotIn("관심사 창업 AI", personalized["search_query"])
        self.assertNotIn("사용자 프로필", personalized["search_query"])
        self.assertIn("지원 가능", personalized["features"]["expanded_terms"])

    def test_it_profile_job_query_does_not_add_it_job_terms_to_query(self):
        profile = {"college": "창의융합대학", "track": "AI응용학과", "grade": "4학년"}

        terms = search_service._profile_search_terms(profile, "내 학과랑 맞는 채용공고 추천해줘")

        self.assertIn("AI응용학과", terms)
        self.assertNotIn("소프트웨어", terms)
        self.assertNotIn("개발", terms)
        self.assertNotIn("백엔드", terms)
        self.assertNotIn("프론트엔드", terms)
        self.assertNotIn("인공지능", terms)

    def test_it_profile_non_job_query_does_not_add_it_job_terms(self):
        profile = {"college": "창의융합대학", "track": "AI응용학과", "grade": "4학년"}

        terms = search_service._profile_search_terms(profile, "내 학과에 맞는 장학금 추천해줘")

        self.assertIn("AI응용학과", terms)
        self.assertNotIn("백엔드", terms)
        self.assertNotIn("프론트엔드", terms)

    def test_it_student_job_rerank_boosts_job_notice_and_penalizes_faculty_hiring(self):
        route = {
            "original_query": "내 학과랑 맞는 채용공고 추천해줘",
            "search_query": "내 학과랑 맞는 채용공고 추천해줘 AI응용학과 4학년",
            "intent": "conditional",
            "features": {},
            "personalization": {
                "applied": True,
                "profile_summary": "단과대학: 창의융합대학; 학과/트랙: AI응용학과; 학년: 4학년",
            },
        }
        job_notice = {
            "title": "[교수님 추천채용] IT 컨설턴트 신입 채용",
            "category": "취업/채용",
            "content": "AI 데이터 소프트웨어 개발 직무. 졸업예정자 지원 가능.",
        }
        faculty_notice = {
            "title": "2026학년도 2학기 전임교원 초빙 공고",
            "category": "취업/채용",
            "content": "컴퓨터공학 AI 소프트웨어 데이터 분야 교수 초빙 공고.",
        }

        job_bonus, job_parts = search_service._score_query_features(job_notice, route)
        faculty_bonus, faculty_parts = search_service._score_query_features(faculty_notice, route)

        self.assertGreater(job_bonus, 0)
        self.assertIn("it_job_keyword", job_parts)
        self.assertIn("student_job_keyword", job_parts)
        self.assertLess(faculty_bonus, job_bonus)
        self.assertIn("academic_hiring_penalty", faculty_parts)

    def test_disabled_query_routing_uses_baseline_candidates_without_router(self):
        baseline = [{"title": "공지", "score": 0.7, "url": "https://example.test/notice"}]

        with (
            mock.patch.object(search_service, "QUERY_ROUTING_ENABLED", False),
            mock.patch.object(search_service, "_route_query_with_gemini") as router,
            mock.patch.object(search_service, "_hybrid_candidate_search", return_value=(baseline, {})) as hybrid,
        ):
            search_query, candidates, diagnostics = search_service._routed_candidates_before_feature_rerank(
                query="장학금 마감 언제야",
                top_k=5,
                alpha=0.5,
                category_filter=None,
                candidate_k=20,
                profile=None,
            )

        self.assertEqual(search_query, "장학금 마감 언제야")
        self.assertEqual(candidates, baseline)
        self.assertEqual(diagnostics["routing_action"], "disabled")
        self.assertEqual(diagnostics["route"]["method"], "disabled")
        router.assert_not_called()
        hybrid.assert_called_once()

    def test_disabled_query_routing_still_applies_personalization(self):
        baseline = [{"title": "공지", "score": 0.7, "url": "https://example.test/notice"}]
        profile = {"track": "컴퓨터공학부", "grade": "3학년", "interests": ["취업/채용"]}

        with (
            mock.patch.object(search_service, "QUERY_ROUTING_ENABLED", False),
            mock.patch.object(search_service, "_route_query_with_gemini") as router,
            mock.patch.object(search_service, "_hybrid_candidate_search", return_value=(baseline, {})) as hybrid,
        ):
            search_query, candidates, diagnostics = search_service._routed_candidates_before_feature_rerank(
                query="내 학과에 맞는 채용 공고 찾아줘",
                top_k=5,
                alpha=0.5,
                category_filter=None,
                candidate_k=20,
                profile=profile,
            )

        self.assertIn("컴퓨터공학부", search_query)
        self.assertIn("3학년", search_query)
        self.assertEqual(candidates, baseline)
        self.assertEqual(diagnostics["routing_action"], "disabled_personalized_query")
        self.assertTrue(diagnostics["route"]["personalization"]["applied"])
        router.assert_not_called()
        hybrid.assert_called_once()


if __name__ == "__main__":
    unittest.main()
