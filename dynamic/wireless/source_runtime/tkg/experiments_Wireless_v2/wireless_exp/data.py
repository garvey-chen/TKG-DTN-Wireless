"""Strict scenario loading; labels are kept separate from policy observations."""

import csv
import hashlib
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from . import ROOT


@dataclass(frozen=True)
class User:
    user_id: str
    x: float
    y: float
    z: float
    cqi: int
    request: str
    ground_truth: str

    def observation(self) -> dict:
        """Only information available to the decision method (no answer label)."""
        return {
            "user_id": self.user_id,
            "location": f"({self.x}, {self.y}, {self.z})",
            "request": self.request,
            "cqi": self.cqi,
        }

    def as_dict(self) -> dict:
        return asdict(self)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_users(path: Path, limit: int | None = None) -> list[User]:
    required = {"RX_ID", "X", "Y", "Z", "CQI", "User_Request", "Request_Label"}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {required - set(reader.fieldnames or [])}")
        rows = list(reader)
    users, seen = [], set()
    for row in rows:
        cqi = float(row["CQI"])
        xyz = [float(row[k]) for k in ("X", "Y", "Z")]
        if not cqi.is_integer() or not 1 <= cqi <= 15 or not all(map(math.isfinite, xyz)):
            raise ValueError(f"Invalid coordinates/CQI for user {row['RX_ID']}")
        uid = row["RX_ID"].strip()
        if not uid or uid in seen or row["Request_Label"] not in ("eMBB", "URLLC"):
            raise ValueError(f"Invalid/duplicate user or label: {uid}")
        if not row["User_Request"].strip():
            raise ValueError(f"Empty request for {uid}")
        seen.add(uid)
        users.append(User(uid, *xyz, int(cqi), row["User_Request"], row["Request_Label"]))
    if limit is not None and (limit < 1 or limit > len(users)):
        raise ValueError(f"Requested {limit} users; file contains {len(users)}")
    return users if limit is None else users[:limit]


def scenario_path(name: str) -> Path:
    if name not in ("center", "north", "south"):
        raise ValueError(f"Unknown scenario: {name}")
    return ROOT / "data" / "scenarios" / f"{name}.csv"
