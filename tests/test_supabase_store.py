import unittest
from unittest import mock

import psycopg

from crawling import supabase_store


class SupabaseStoreConnectionTests(unittest.TestCase):
    def test_connect_retries_transient_operational_error(self):
        connection = object()
        connect = mock.Mock(
            side_effect=[
                psycopg.OperationalError("timeout"),
                psycopg.OperationalError("timeout"),
                connection,
            ]
        )

        with (
            mock.patch.object(supabase_store.psycopg, "connect", connect),
            mock.patch.object(supabase_store.time, "sleep") as sleep,
            mock.patch.dict(
                "os.environ",
                {
                    "SUPABASE_DB_URL": "postgresql://example",
                    "SUPABASE_DB_CONNECT_RETRIES": "3",
                    "SUPABASE_DB_CONNECT_TIMEOUT": "7",
                    "SUPABASE_DB_CONNECT_RETRY_DELAY": "0.1",
                },
                clear=False,
            ),
        ):
            self.assertIs(supabase_store._connect(), connection)

        self.assertEqual(connect.call_count, 3)
        connect.assert_called_with(
            "postgresql://example",
            autocommit=False,
            connect_timeout=7,
            prepare_threshold=None,
        )
        self.assertEqual(sleep.call_count, 2)

    def test_connect_raises_after_retry_limit(self):
        connect = mock.Mock(side_effect=psycopg.OperationalError("timeout"))

        with (
            mock.patch.object(supabase_store.psycopg, "connect", connect),
            mock.patch.object(supabase_store.time, "sleep"),
            mock.patch.dict(
                "os.environ",
                {
                    "SUPABASE_DB_URL": "postgresql://example",
                    "SUPABASE_DB_CONNECT_RETRIES": "2",
                    "SUPABASE_DB_CONNECT_RETRY_DELAY": "0",
                },
                clear=False,
            ),
        ):
            with self.assertRaisesRegex(psycopg.OperationalError, "timeout"):
                supabase_store._connect()

        self.assertEqual(connect.call_count, 2)


if __name__ == "__main__":
    unittest.main()
