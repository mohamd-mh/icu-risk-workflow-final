"""Local SQLite workflow storage for the ICU review queue information system."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = Path(os.getenv("ICU_REVIEW_DB_PATH", BASE_DIR / "data" / "software_system.db"))
ACTIVE_STATUSES = ("New", "In Review", "Reviewed", "Needs Follow-up", "Closed")
STATUSES = ["New", "In Review", "Reviewed", "Needs Follow-up", "Closed", "Archived"]
PRIORITIES = ["Low", "Watch", "Urgent"]


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS review_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT NOT NULL,
                icustay_id TEXT,
                subject_id TEXT,
                hadm_id TEXT,
                manual_case_id INTEGER,
                local_review_reference TEXT DEFAULT '',
                assigned_reviewer TEXT DEFAULT '',
                status TEXT DEFAULT 'New',
                priority TEXT DEFAULT 'Low',
                ai_risk_level TEXT,
                ai_probability REAL,
                data_quality TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS manual_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                local_review_reference TEXT DEFAULT '',
                gender TEXT,
                age REAL,
                admission_type TEXT,
                heart_rate_avg REAL,
                heart_rate_max REAL,
                systolic_bp_avg REAL,
                respiratory_rate_avg REAL,
                temperature_avg REAL,
                spo2_avg REAL,
                creatinine_max REAL,
                glucose_max REAL,
                wbc_max REAL,
                hemoglobin_min REAL,
                ai_risk_level TEXT,
                ai_probability REAL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS review_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                review_item_id INTEGER NOT NULL,
                reviewer_name TEXT,
                note_text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY(review_item_id) REFERENCES review_items(id)
            );
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                entity_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                details TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        ensure_column(conn, "review_items", "local_review_reference", "TEXT DEFAULT ''")
        ensure_column(conn, "manual_cases", "local_review_reference", "TEXT DEFAULT ''")


def ensure_column(conn: sqlite3.Connection, table_name: str, column_name: str, column_definition: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table_name})").fetchall()}
    if column_name not in columns:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_definition}")


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row else None


def add_audit_event(entity_type: str, entity_id: int, action: str, details: dict[str, Any] | str | None = None) -> None:
    detail_text = json.dumps(details, sort_keys=True) if isinstance(details, dict) else (details or "")
    with connect() as conn:
        conn.execute(
            "INSERT INTO audit_events(entity_type, entity_id, action, details, created_at) VALUES (?, ?, ?, ?, ?)",
            (entity_type, entity_id, action, detail_text, now()),
        )


def default_priority_for_risk(risk_level: str | None) -> str:
    if risk_level == "High":
        return "Urgent"
    if risk_level == "Medium":
        return "Watch"
    return "Low"


def get_active_review_item_for_icustay(icustay_id: str) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute(
            f"SELECT * FROM review_items WHERE source_type='dataset_case' AND icustay_id=? AND status IN ({','.join('?' for _ in ACTIVE_STATUSES)}) ORDER BY id DESC LIMIT 1",
            (str(icustay_id), *ACTIVE_STATUSES),
        ).fetchone()
    return row_to_dict(row)


def normalize_priority(priority: str | None, fallback_risk_level: str | None = None) -> str:
    if priority in PRIORITIES:
        return priority
    return default_priority_for_risk(fallback_risk_level)


def create_dataset_review_item(case: dict[str, Any], assigned_reviewer: str = "", suggested_priority: str | None = None) -> tuple[dict[str, Any], bool]:
    existing = get_active_review_item_for_icustay(str(case.get("icustay_id")))
    if existing:
        return existing, False
    ai_result = case.get("ai_prediction_result", {})
    timestamp = now()
    priority = normalize_priority(suggested_priority, ai_result.get("ai_risk_level"))
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO review_items(source_type, icustay_id, subject_id, hadm_id, assigned_reviewer, status, priority, ai_risk_level, ai_probability, data_quality, created_at, updated_at)
            VALUES ('dataset_case', ?, ?, ?, ?, 'New', ?, ?, ?, ?, ?, ?)
            """,
            (
                str(case.get("icustay_id")), str(case.get("subject_id")), str(case.get("hadm_id")), assigned_reviewer,
                priority, ai_result.get("ai_risk_level"), ai_result.get("risk_probability"), case.get("data_quality"), timestamp, timestamp,
            ),
        )
        item_id = int(cur.lastrowid)
    add_audit_event("review_item", item_id, "Created review item from dataset case", {"icustay_id": case.get("icustay_id")})
    return get_review_item(item_id) or {"id": item_id}, True


def create_manual_case(case_values: dict[str, Any], ai_result: dict[str, Any]) -> dict[str, Any]:
    timestamp = now()
    fields = [
        "local_review_reference", "gender", "age", "admission_type", "heart_rate_avg", "heart_rate_max", "systolic_bp_avg",
        "respiratory_rate_avg", "temperature_avg", "spo2_avg", "creatinine_max", "glucose_max", "wbc_max", "hemoglobin_min",
    ]
    with connect() as conn:
        cur = conn.execute(
            f"INSERT INTO manual_cases({', '.join(fields)}, ai_risk_level, ai_probability, created_at, updated_at) VALUES ({', '.join('?' for _ in fields)}, ?, ?, ?, ?)",
            tuple(case_values.get(field) for field in fields) + (ai_result.get("ai_risk_level"), ai_result.get("risk_probability"), timestamp, timestamp),
        )
        manual_id = int(cur.lastrowid)
    add_audit_event(
        "manual_case",
        manual_id,
        "Created manual case submission",
        {"ai_risk_level": ai_result.get("ai_risk_level"), "local_review_reference": case_values.get("local_review_reference", "")},
    )
    return get_manual_case(manual_id) or {"id": manual_id}


def create_manual_review_item(manual_case: dict[str, Any], data_quality: str = "Manual", suggested_priority: str | None = None) -> dict[str, Any]:
    timestamp = now()
    priority = normalize_priority(suggested_priority, manual_case.get("ai_risk_level"))
    with connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO review_items(source_type, manual_case_id, local_review_reference, status, priority, ai_risk_level, ai_probability, data_quality, created_at, updated_at)
            VALUES ('manual_case', ?, ?, 'New', ?, ?, ?, ?, ?, ?)
            """,
            (
                manual_case.get("id"),
                manual_case.get("local_review_reference", ""),
                priority,
                manual_case.get("ai_risk_level"),
                manual_case.get("ai_probability"),
                data_quality,
                timestamp,
                timestamp,
            ),
        )
        item_id = int(cur.lastrowid)
    add_audit_event("review_item", item_id, "Created review item from manual case", {"manual_case_id": manual_case.get("id")})
    return get_review_item(item_id) or {"id": item_id}


def list_review_items(filters: dict[str, str] | None = None) -> list[dict[str, Any]]:
    filters = filters or {}
    clauses: list[str] = []
    params: list[Any] = []
    status_filter = filters.get("status")
    if status_filter == "__active__":
        clauses.append("ri.status != 'Archived'")
    elif status_filter and status_filter != "All":
        clauses.append("ri.status = ?")
        params.append(status_filter)
    for field in ["priority", "ai_risk_level"]:
        if filters.get(field):
            clauses.append(f"ri.{field} = ?")
            params.append(filters[field])
    if filters.get("source_type"):
        clauses.append("ri.source_type = ?")
        params.append(filters["source_type"])
    if filters.get("assigned_reviewer"):
        clauses.append("ri.assigned_reviewer = ?")
        params.append(filters["assigned_reviewer"])
    search = (filters.get("search") or "").strip()
    if search:
        clauses.append("(ri.icustay_id LIKE ? OR ri.assigned_reviewer LIKE ? OR ri.subject_id LIKE ? OR ri.hadm_id LIKE ? OR ri.local_review_reference LIKE ?)")
        like = f"%{search}%"
        params.extend([like, like, like, like, like])
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    query = f"""
        SELECT ri.*, rn.note_text AS latest_note, rn.reviewer_name AS latest_note_reviewer, rn.created_at AS latest_note_at
        FROM review_items ri
        LEFT JOIN review_notes rn ON rn.id = (
            SELECT id FROM review_notes WHERE review_item_id = ri.id ORDER BY created_at DESC, id DESC LIMIT 1
        )
        {where}
        ORDER BY
            CASE ri.priority WHEN 'Urgent' THEN 0 WHEN 'Watch' THEN 1 ELSE 2 END,
            CASE ri.ai_risk_level WHEN 'High' THEN 0 WHEN 'Medium' THEN 1 WHEN 'Low' THEN 2 ELSE 3 END,
            CASE ri.status WHEN 'Needs Follow-up' THEN 0 WHEN 'New' THEN 1 WHEN 'In Review' THEN 2 WHEN 'Reviewed' THEN 3 WHEN 'Closed' THEN 4 ELSE 5 END,
            ri.updated_at DESC,
            ri.id DESC
    """
    with connect() as conn:
        return [dict(row) for row in conn.execute(query, params).fetchall()]


def get_review_item(item_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM review_items WHERE id=?", (item_id,)).fetchone()
    return row_to_dict(row)


def get_manual_case(case_id: int) -> dict[str, Any] | None:
    with connect() as conn:
        row = conn.execute("SELECT * FROM manual_cases WHERE id=?", (case_id,)).fetchone()
    return row_to_dict(row)


def update_review_item(item_id: int, status: str, priority: str, assigned_reviewer: str) -> None:
    if status not in STATUSES:
        status = "New"
    if priority not in PRIORITIES:
        priority = "Low"
    with connect() as conn:
        conn.execute(
            "UPDATE review_items SET status=?, priority=?, assigned_reviewer=?, updated_at=? WHERE id=?",
            (status, priority, assigned_reviewer.strip(), now(), item_id),
        )
    add_audit_event("review_item", item_id, "Updated review item", {"status": status, "priority": priority, "assigned_reviewer": assigned_reviewer})


def add_review_note(item_id: int, reviewer_name: str, note_text: str) -> None:
    timestamp = now()
    with connect() as conn:
        conn.execute(
            "INSERT INTO review_notes(review_item_id, reviewer_name, note_text, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (item_id, reviewer_name.strip(), note_text.strip(), timestamp, timestamp),
        )
        conn.execute("UPDATE review_items SET updated_at=? WHERE id=?", (timestamp, item_id))
    add_audit_event("review_item", item_id, "Added review note", {"reviewer_name": reviewer_name})


def archive_review_item(item_id: int) -> None:
    with connect() as conn:
        conn.execute("UPDATE review_items SET status='Archived', updated_at=? WHERE id=?", (now(), item_id))
    add_audit_event("review_item", item_id, "Archived review item")


def delete_review_item(item_id: int) -> None:
    archive_review_item(item_id)
    add_audit_event("review_item", item_id, "Delete requested; archived instead")


def get_review_notes(item_id: int) -> list[dict[str, Any]]:
    with connect() as conn:
        return [dict(row) for row in conn.execute("SELECT * FROM review_notes WHERE review_item_id=? ORDER BY created_at DESC, id DESC", (item_id,)).fetchall()]


def get_audit_events(entity_type: str | None = None, entity_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
    clauses = []
    params: list[Any] = []
    if entity_type:
        clauses.append("entity_type=?")
        params.append(entity_type)
    if entity_id is not None:
        clauses.append("entity_id=?")
        params.append(entity_id)
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    params.append(limit)
    with connect() as conn:
        return [dict(row) for row in conn.execute(f"SELECT * FROM audit_events {where} ORDER BY created_at DESC, id DESC LIMIT ?", params).fetchall()]


def get_workflow_stats() -> dict[str, Any]:
    with connect() as conn:
        total_items = conn.execute("SELECT COUNT(*) FROM review_items").fetchone()[0]
        manual_cases = conn.execute("SELECT COUNT(*) FROM manual_cases").fetchone()[0]
        urgent = conn.execute("SELECT COUNT(*) FROM review_items WHERE priority='Urgent' AND status!='Archived'").fetchone()[0]
        high_unreviewed = conn.execute("SELECT COUNT(*) FROM review_items WHERE ai_risk_level='High' AND status IN ('New','In Review','Needs Follow-up')").fetchone()[0]
        status_counts = {status: 0 for status in STATUSES}
        for row in conn.execute("SELECT status, COUNT(*) AS count FROM review_items GROUP BY status").fetchall():
            status_counts[row["status"]] = row["count"]
    return {"total_items": total_items, "manual_cases": manual_cases, "urgent": urgent, "high_unreviewed": high_unreviewed, "status_counts": status_counts}


init_db()
