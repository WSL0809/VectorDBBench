from typing import TypedDict

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class DingoDBConfigDict(TypedDict):
    addrs: str
    index_name: str


class DingoDBConfig(DBConfig):
    addrs: SecretStr
    index_name: str = "vectordb_bench_index"

    def to_dict(self) -> DingoDBConfigDict:
        return {
            "addrs": self.addrs.get_secret_value() if isinstance(self.addrs, SecretStr) else self.addrs,
            "index_name": self.index_name,
        }


class DingoDBCaseConfig(BaseModel, DBCaseConfig):
    metric_type: MetricType | None = None

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        # Provide sensible defaults to improve recall for approximate search
        # efSearch mainly affects HNSW recall; other fields are safe no-ops when unsupported
        return {
            "efSearch": 256,
            "recallNum": 1000,
            "parallelOnQueries": 0,
        }
