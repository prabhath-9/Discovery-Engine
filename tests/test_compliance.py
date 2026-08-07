from __future__ import annotations

from src.compliance.erasure import TOMBSTONE_KEY, DeletionReceipt, erase_user


class _FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, query: str, params: tuple[object, ...] = ()) -> None:
        self.executed.append((query, params))


class _FakePGConnection:
    def __init__(self) -> None:
        self._cursor = _FakeCursor()
        self.committed = False

    def cursor(self) -> _FakeCursor:
        return self._cursor

    def commit(self) -> None:
        self.committed = True


class _FakeRedisClient:
    def __init__(self) -> None:
        self.deleted: list[str] = []
        self.sets: dict[str, set[str]] = {}

    def delete(self, *keys: str) -> int:
        self.deleted.extend(keys)
        return len(keys)

    def sadd(self, key: str, *values: str) -> int:
        self.sets.setdefault(key, set()).update(values)
        return len(values)


class _MockIndexHandle:
    def __init__(self) -> None:
        self.removed: list[str] = []

    def remove(self, user_id: str) -> None:
        self.removed.append(user_id)


def test_erase_user_removes_user_from_all_stores_and_returns_receipt() -> None:
    pg_conn = _FakePGConnection()
    redis_client = _FakeRedisClient()
    index_handle = _MockIndexHandle()

    receipt = erase_user("user-1", pg_conn, redis_client, index_handle)

    assert isinstance(receipt, DeletionReceipt)
    assert receipt.user_id == "user-1"
    assert receipt.receipt_id

    queries = [q for q, _ in pg_conn.cursor().executed]
    assert any("DELETE FROM users" in q for q in queries)
    assert any("UPDATE recommendation_log" in q for q in queries)
    assert pg_conn.committed is True

    assert "sess:user-1" in redis_client.deleted
    assert "feat:u:user-1" in redis_client.deleted
    assert "user-1" in redis_client.sets[TOMBSTONE_KEY]

    assert index_handle.removed == ["user-1"]


def test_erase_user_returns_unique_receipt_ids() -> None:
    pg_conn = _FakePGConnection()
    redis_client = _FakeRedisClient()
    index_handle = _MockIndexHandle()

    first = erase_user("user-1", pg_conn, redis_client, index_handle)
    second = erase_user("user-2", pg_conn, redis_client, index_handle)

    assert first.receipt_id != second.receipt_id
