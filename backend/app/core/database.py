import base64
from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import httpx

from app.core.config import settings
from app.core.errors import DatabaseConnectionError, DatabaseNotConfiguredError
from app.core.logging import logger


class TursoRow:
    """Row object behaving like sqlite3.Row supporting key, case-insensitive key, and index access."""

    def __init__(self, cols: List[str], values: List[Any]):
        self._cols = cols
        self._values = values
        self._mapping: Dict[str, Any] = dict(zip(cols, values))

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, int):
            return self._values[key]
        if isinstance(key, str):
            if key in self._mapping:
                return self._mapping[key]
            key_lower = key.lower()
            for k, v in self._mapping.items():
                if k.lower() == key_lower:
                    return v
            raise KeyError(key)
        raise TypeError(f"Row indices must be integers or strings, not {type(key)}")

    def get(self, key: str, default: Any = None) -> Any:
        if key in self._mapping:
            return self._mapping[key]
        key_lower = key.lower()
        for k, v in self._mapping.items():
            if k.lower() == key_lower:
                return v
        return default

    def keys(self) -> List[str]:
        return list(self._cols)

    def values(self) -> List[Any]:
        return list(self._values)

    def items(self):
        return self._mapping.items()

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, str):
            if key in self._mapping:
                return True
            key_lower = key.lower()
            return any(k.lower() == key_lower for k in self._mapping)
        return False

    def __repr__(self) -> str:
        return f"<TursoRow {self._mapping}>"


class TursoCursor:
    """Cursor object representing query result set compatible with sqlite3.Cursor."""

    def __init__(
        self,
        cols: List[str],
        rows: List[TursoRow],
        rowcount: int = -1,
        lastrowid: Optional[int] = None,
    ):
        self._cols = cols
        self._rows = rows
        self._rowcount = rowcount
        self._lastrowid = lastrowid
        self._idx = 0

    @property
    def rowcount(self) -> int:
        return self._rowcount

    @property
    def lastrowid(self) -> Optional[int]:
        return self._lastrowid

    @property
    def description(self) -> Tuple[Any, ...]:
        return tuple((c, None, None, None, None, None, None) for c in self._cols)

    def fetchone(self) -> Optional[TursoRow]:
        if self._idx < len(self._rows):
            r = self._rows[self._idx]
            self._idx += 1
            return r
        return None

    def fetchall(self) -> List[TursoRow]:
        remaining = self._rows[self._idx :]
        self._idx = len(self._rows)
        return remaining

    def __iter__(self):
        return self

    def __next__(self) -> TursoRow:
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row


class TursoHttpConnection:
    """
    HTTP pipeline client adapter for Turso / libSQL Cloud.
    Enables serverless-friendly SQLite database queries over HTTPS without native C-extensions.
    Never logs auth tokens or credentials.
    """

    def __init__(
        self,
        database_url: str,
        auth_token: str,
        timeout_seconds: float = 30.0,
        client: Optional[httpx.Client] = None,
    ):
        self.auth_token = auth_token
        clean_url = database_url.strip().rstrip("/")
        if clean_url.startswith("libsql://"):
            clean_url = "https://" + clean_url[len("libsql://") :]
        elif not clean_url.startswith("http://") and not clean_url.startswith("https://"):
            clean_url = f"https://{clean_url}"
        self.pipeline_url = f"{clean_url}/v2/pipeline"
        self.timeout = timeout_seconds
        self._custom_client = client

    @staticmethod
    def _serialize_value(val: Any) -> Dict[str, Any]:
        if val is None:
            return {"type": "null"}
        if isinstance(val, bool):
            return {"type": "integer", "value": "1" if val else "0"}
        if isinstance(val, int):
            return {"type": "integer", "value": str(val)}
        if isinstance(val, float):
            return {"type": "float", "value": val}
        if isinstance(val, bytes):
            return {"type": "blob", "base64": base64.b64encode(val).decode("ascii")}
        if isinstance(val, str):
            return {"type": "text", "value": val}
        return {"type": "text", "value": str(val)}

    @staticmethod
    def _deserialize_cell(cell: Any) -> Any:
        if isinstance(cell, dict):
            cell_type = cell.get("type")
            if cell_type == "null":
                return None
            if cell_type == "integer":
                val = cell.get("value")
                return int(val) if val is not None else None
            if cell_type == "float":
                val = cell.get("value")
                return float(val) if val is not None else None
            if cell_type == "text":
                return cell.get("value")
            if cell_type == "blob":
                b64_val = cell.get("base64")
                if b64_val is not None:
                    return base64.b64decode(b64_val)
                val = cell.get("value")
                if isinstance(val, str):
                    return base64.b64decode(val)
                return val
            return cell.get("value")
        return cell

    def _post_pipeline(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }
        try:
            if self._custom_client:
                resp = self._custom_client.post(self.pipeline_url, json=payload, headers=headers)
            else:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(self.pipeline_url, json=payload, headers=headers)
        except Exception as exc:
            raise DatabaseConnectionError(f"Failed to connect to Turso database: {str(exc)}") from exc

        if resp.status_code in (401, 403):
            raise DatabaseNotConfiguredError("Turso authentication failed. Please check TURSO_AUTH_TOKEN.")
        if resp.status_code != 200:
            raise DatabaseConnectionError(f"Turso pipeline HTTP error {resp.status_code}: {resp.text[:200]}")

        return resp.json()

    def execute(self, sql: str, params: Any = ()) -> TursoCursor:
        clean_sql = sql.strip().upper()
        # Handle SQLite-only file locking pragmas gracefully as no-ops
        if clean_sql.startswith("PRAGMA JOURNAL_MODE") or clean_sql.startswith("PRAGMA SYNCHRONOUS"):
            return TursoCursor(cols=[], rows=[], rowcount=0)

        serialized_args = [
            self._serialize_value(p)
            for p in (params if isinstance(params, (list, tuple)) else [params])
        ]
        request_body = {
            "requests": [
                {
                    "type": "execute",
                    "stmt": {
                        "sql": sql,
                        "args": serialized_args,
                    },
                },
                {"type": "close"},
            ]
        }

        resp_json = self._post_pipeline(request_body)
        results = resp_json.get("results", [])
        if not results:
            return TursoCursor(cols=[], rows=[], rowcount=0)

        first_res = results[0]
        if first_res.get("type") == "error":
            err_msg = first_res.get("error", {}).get("message", "Unknown query error")
            raise DatabaseConnectionError(f"Turso query error: {err_msg}")

        stmt_result = first_res.get("response", {}).get("result", {})
        cols = [
            c.get("name", "") if isinstance(c, dict) else str(c)
            for c in stmt_result.get("cols", [])
        ]
        raw_rows = stmt_result.get("rows", [])

        rows: List[TursoRow] = []
        for r in raw_rows:
            parsed_row = [self._deserialize_cell(cell) for cell in r]
            rows.append(TursoRow(cols=cols, values=parsed_row))

        affected = stmt_result.get("affected_row_count", 0)
        last_id = stmt_result.get("last_insert_rowid")
        return TursoCursor(cols=cols, rows=rows, rowcount=affected, lastrowid=last_id)

    def executemany(self, sql: str, seq_of_params: Any) -> TursoCursor:
        param_list = list(seq_of_params)
        if not param_list:
            return TursoCursor(cols=[], rows=[], rowcount=0)

        requests = []
        for params in param_list:
            serialized_args = [
                self._serialize_value(p)
                for p in (params if isinstance(params, (list, tuple)) else [params])
            ]
            requests.append({
                "type": "execute",
                "stmt": {
                    "sql": sql,
                    "args": serialized_args,
                },
            })
        requests.append({"type": "close"})

        request_body = {"requests": requests}
        resp_json = self._post_pipeline(request_body)
        results = resp_json.get("results", [])
        total_affected = 0
        for r in results:
            if r.get("type") == "error":
                err_msg = r.get("error", {}).get("message", "Unknown batch error")
                raise DatabaseConnectionError(f"Turso batch query error: {err_msg}")
            if r.get("type") == "ok":
                stmt_result = r.get("response", {}).get("result", {})
                total_affected += stmt_result.get("affected_row_count", 0)

        return TursoCursor(cols=[], rows=[], rowcount=total_affected)

    def commit(self) -> None:
        pass

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


@contextmanager
def get_db_connection(
    db_path: Optional[Path] = None,
    provider: Optional[str] = None,
    client: Optional[httpx.Client] = None,
) -> Generator[Any, None, None]:
    """
    Provides a managed database connection for either local SQLite or Turso libSQL cloud.
    Commits automatically on normal block exit, rolls back on exceptions, and closes connection.
    """
    active_provider = provider
    if active_provider is None:
        if db_path is not None and db_path != settings.database_path:
            active_provider = "sqlite"
        else:
            active_provider = settings.DATABASE_PROVIDER.lower().strip()

    if active_provider == "turso":
        if not settings.TURSO_DATABASE_URL or not settings.TURSO_AUTH_TOKEN:
            raise DatabaseNotConfiguredError(
                "Database is not configured. Please set TURSO_DATABASE_URL and TURSO_AUTH_TOKEN in your environment or backend/.env."
            )
        conn = TursoHttpConnection(
            database_url=settings.TURSO_DATABASE_URL,
            auth_token=settings.TURSO_AUTH_TOKEN,
            client=client,
        )
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    else:
        path = db_path or settings.database_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sqlite_conn = sqlite3.connect(str(path), timeout=15.0)
        sqlite_conn.row_factory = sqlite3.Row
        try:
            sqlite_conn.execute("PRAGMA journal_mode=WAL;")
            sqlite_conn.execute("PRAGMA synchronous=NORMAL;")
            sqlite_conn.execute("PRAGMA foreign_keys=ON;")
            yield sqlite_conn
            sqlite_conn.commit()
        except Exception:
            sqlite_conn.rollback()
            raise
        finally:
            sqlite_conn.close()
