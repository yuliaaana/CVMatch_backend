import unittest
from unittest.mock import patch

from cvmatch import fetch_jobs, infer_job_categories


class InferJobCategoriesTests(unittest.TestCase):
    def test_detects_multiple_resume_categories(self):
        categories = infer_job_categories(
            "Data Engineer with Python, Spark, Airflow and machine learning experience."
        )

        self.assertEqual(
            categories,
            ["Python", "Data Science / ML", "Data Engineer"],
        )

    def test_falls_back_to_all_categories_when_no_category_is_detected(self):
        self.assertEqual(infer_job_categories("Experienced project coordinator"), ["Усі категорії"])

    def test_fetch_jobs_queries_the_selected_site_categories(self):
        with (
            patch("cvmatch.fetch_dou_jobs", return_value=[]) as fetch_dou,
            patch("cvmatch.fetch_djinni_jobs", return_value=[]) as fetch_djinni,
        ):
            jobs, errors = fetch_jobs(categories=["Python"], timeout=3)

        self.assertEqual(jobs, [])
        self.assertEqual(errors, [])
        fetch_dou.assert_called_once_with(category="Python", timeout=3)
        fetch_djinni.assert_called_once_with(keyword="python", timeout=3)


if __name__ == "__main__":
    unittest.main()
