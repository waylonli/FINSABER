from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ArtifactWindow:
    requested_start: date
    requested_end: date
    effective_start: date
    effective_end: date

    def to_dict(self) -> dict[str, str]:
        return {
            "requested_start": self.requested_start.isoformat(),
            "requested_end": self.requested_end.isoformat(),
            "effective_start": self.effective_start.isoformat(),
            "effective_end": self.effective_end.isoformat(),
        }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _safe_path_component(value: Any) -> str:
    return str(value).replace("/", "_").replace("\\", "_").replace(":", "_")


def _loader_summary(loader: Any) -> dict[str, Any] | None:
    if loader is None:
        return None

    summary: dict[str, Any] = {
        "class_name": loader.__class__.__name__,
        "module": loader.__class__.__module__,
        "source_kind": getattr(loader, "source_kind", None),
        "filing_payload_kind": getattr(loader, "filing_payload_kind", None),
    }

    for attr_name in (
        "root",
        "start_date",
        "end_date",
        "tickers",
        "modalities",
        "price_field",
        "filing_merge_policy",
        "failure_mode",
    ):
        if hasattr(loader, attr_name):
            summary[attr_name] = _json_safe(getattr(loader, attr_name))

    if hasattr(loader, "_section_map"):
        summary["section_map"] = _json_safe(getattr(loader, "_section_map"))

    if hasattr(loader, "base_loader"):
        # Nested loaders are summarized recursively so we can see the full
        # data path without dumping the in-memory dataset itself.
        summary["base_loader"] = _loader_summary(getattr(loader, "base_loader"))

    return summary


class FinMemArtifactWriter:
    def __init__(
        self,
        *,
        artifact_config: Mapping[str, Any] | None,
        symbol: str,
        config_path: str,
        resolved_strategy_params: Mapping[str, Any],
        finmem_config: Mapping[str, Any],
        requested_train_window: ArtifactWindow,
        requested_test_window: ArtifactWindow,
        input_data_loader: Any,
        runtime_market_data: Any,
        filing_options: Mapping[str, Any],
    ) -> None:
        config = dict(artifact_config or {})
        normalized_strategy_params = {
            key: value
            for key, value in dict(resolved_strategy_params).items()
            if key != "artifact_config"
        }
        self.enabled = bool(config.get("enabled", False))
        self.root = Path(config.get("root", "backtest/output/finmem_artifacts")).expanduser().resolve()
        self.save_agent_checkpoint = bool(config.get("save_agent_checkpoint", True))
        self.save_environment_checkpoint = bool(
            config.get("save_environment_checkpoint", True)
        )
        self.save_reflections = bool(config.get("save_reflections", True))
        self.save_query_trace = bool(config.get("save_query_trace", True))
        self.save_llm_trace = bool(config.get("save_llm_trace", True))

        self.symbol = symbol
        self.config_path = str(Path(config_path).resolve())
        self.requested_train_window = requested_train_window
        self.requested_test_window = requested_test_window
        self.config_key = self._build_config_key(
            config_path=self.config_path,
            finmem_config=finmem_config,
            resolved_strategy_params=normalized_strategy_params,
        )
        self._manifest = {
            "symbol": symbol,
            "config_path": self.config_path,
            "artifact_config": {
                "enabled": self.enabled,
                "root": str(self.root),
                "save_agent_checkpoint": self.save_agent_checkpoint,
                "save_environment_checkpoint": self.save_environment_checkpoint,
                "save_reflections": self.save_reflections,
                "save_query_trace": self.save_query_trace,
                "save_llm_trace": self.save_llm_trace,
            },
            "windows": {
                "train": requested_train_window.to_dict(),
                "test": requested_test_window.to_dict(),
            },
            "artifact_layout": {
                "window_key": self._window_key(),
                "config_key": self.config_key,
                "ticker_dir": str(self._ticker_dir()),
            },
            "resolved_strategy_params": _json_safe(normalized_strategy_params),
            "filing_options": _json_safe(filing_options),
            "input_data_loader": _loader_summary(input_data_loader),
            "runtime_market_data": _loader_summary(runtime_market_data),
            "finmem_config": _json_safe(finmem_config),
            "artifact_notes": {
                "post_train": "Captured after FinMem training completes.",
                "test_state": (
                    "Captures strategy-local test state only. See test_state/"
                    "snapshot_meta.json for the exact capture stage and reason."
                ),
                "query_trace": "Captures the memory payload selected for reflection.",
                "llm_trace": (
                    "Captures the reflection prompt payload, raw LLM outputs, "
                    "validated output, and returned payload."
                ),
            },
        }

    @staticmethod
    def _build_config_key(
        *,
        config_path: str,
        finmem_config: Mapping[str, Any],
        resolved_strategy_params: Mapping[str, Any],
    ) -> str:
        config_stem = _safe_path_component(Path(config_path).stem)
        fingerprint_source = {
            "config_path": config_path,
            "finmem_config": _json_safe(finmem_config),
            "resolved_strategy_params": _json_safe(resolved_strategy_params),
        }
        fingerprint_json = json.dumps(
            fingerprint_source,
            ensure_ascii=False,
            sort_keys=True,
        )
        fingerprint = hashlib.sha256(fingerprint_json.encode("utf-8")).hexdigest()[:10]
        return f"{config_stem}_{fingerprint}"

    def _window_key(self) -> str:
        return (
            "train_"
            f"{self.requested_train_window.requested_start.isoformat()}_"
            f"{self.requested_train_window.requested_end.isoformat()}__"
            "test_"
            f"{self.requested_test_window.requested_start.isoformat()}_"
            f"{self.requested_test_window.requested_end.isoformat()}"
        )

    def _ticker_dir(self) -> Path:
        return self.root / self._window_key() / self.config_key / self.symbol

    def _phase_dir(self, phase_name: str) -> Path:
        return self._ticker_dir() / phase_name

    def _write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            json.dump(_json_safe(payload), file, ensure_ascii=False, indent=2)

    def _write_jsonl(self, path: Path, rows: list[Mapping[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(_json_safe(row), ensure_ascii=False))
                file.write("\n")

    @staticmethod
    def _normalize_record_date(record_date: Any) -> date:
        if isinstance(record_date, datetime):
            return record_date.date()
        if isinstance(record_date, date):
            return record_date
        if isinstance(record_date, str):
            return datetime.fromisoformat(record_date).date()
        raise TypeError(f"Unsupported reflection key type: {type(record_date)!r}")

    def _write_reflections(
        self,
        *,
        phase_name: str,
        output_name: str,
        reflection_series: Mapping[Any, Any],
        window: ArtifactWindow,
    ) -> None:
        rows: list[dict[str, Any]] = []
        for record_date, payload in sorted(
            reflection_series.items(),
            key=lambda item: self._normalize_record_date(item[0]),
        ):
            normalized_date = self._normalize_record_date(record_date)
            if not (window.effective_start <= normalized_date <= window.effective_end):
                continue
            rows.append(
                {
                    "phase": phase_name,
                    "date": normalized_date,
                    "payload": payload,
                }
            )
        self._write_jsonl(self._ticker_dir() / output_name, rows)

    def _write_agent_series(
        self,
        *,
        phase_name: str,
        output_name: str,
        series: Mapping[Any, Any],
        window: ArtifactWindow,
    ) -> None:
        rows: list[dict[str, Any]] = []
        for record_date, payload in sorted(
            series.items(),
            key=lambda item: self._normalize_record_date(item[0]),
        ):
            normalized_date = self._normalize_record_date(record_date)
            if not (window.effective_start <= normalized_date <= window.effective_end):
                continue
            rows.append(
                {
                    "phase": phase_name,
                    "date": normalized_date,
                    "payload": payload,
                }
            )
        self._write_jsonl(self._ticker_dir() / output_name, rows)

    def _save_phase_checkpoint(
        self,
        *,
        phase_dir: Path,
        agent: Any,
        environment: Any,
    ) -> None:
        phase_dir.mkdir(parents=True, exist_ok=True)
        if self.save_agent_checkpoint:
            agent.save_checkpoint(str(phase_dir), force=True)
        if self.save_environment_checkpoint:
            environment.save_checkpoint(str(phase_dir), force=True)

    def _write_snapshot_metadata(
        self,
        *,
        phase_dir: Path,
        metadata: Mapping[str, Any],
    ) -> None:
        self._write_json(phase_dir / "snapshot_meta.json", metadata)

    def write_manifest(self) -> None:
        if not self.enabled:
            return
        manifest = dict(self._manifest)
        manifest["generated_at"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        self._write_json(self._ticker_dir() / "manifest.json", manifest)

    def save_post_train(self, *, agent: Any, environment: Any) -> None:
        if not self.enabled:
            return

        self.write_manifest()
        phase_dir = self._phase_dir("post_train")
        self._save_phase_checkpoint(
            phase_dir=phase_dir,
            agent=agent,
            environment=environment,
        )
        self._write_snapshot_metadata(
            phase_dir=phase_dir,
            metadata={
                "capture_stage": "post_train",
                "snapshot_reason": "training_completed",
            },
        )
        if self.save_reflections:
            self._write_reflections(
                phase_name="train",
                output_name="train_reflections.jsonl",
                reflection_series=getattr(agent, "reflection_result_series_dict", {}),
                window=self.requested_train_window,
            )
        if self.save_query_trace:
            self._write_agent_series(
                phase_name="train",
                output_name="train_query_trace.jsonl",
                series=getattr(agent, "query_trace_series_dict", {}),
                window=self.requested_train_window,
            )
        if self.save_llm_trace:
            self._write_agent_series(
                phase_name="train",
                output_name="train_llm_trace.jsonl",
                series=getattr(agent, "llm_trace_series_dict", {}),
                window=self.requested_train_window,
            )

    def save_test_state(
        self,
        *,
        agent: Any,
        environment: Any,
        capture_stage: str,
        snapshot_reason: str,
        framework_status: bool | None = None,
    ) -> None:
        if not self.enabled:
            return

        self.write_manifest()
        phase_dir = self._phase_dir("test_state")
        self._save_phase_checkpoint(
            phase_dir=phase_dir,
            agent=agent,
            environment=environment,
        )
        self._write_snapshot_metadata(
            phase_dir=phase_dir,
            metadata={
                "capture_stage": capture_stage,
                "snapshot_reason": snapshot_reason,
                "framework_status": framework_status,
            },
        )
        if self.save_reflections:
            self._write_reflections(
                phase_name="test",
                output_name="test_reflections.jsonl",
                reflection_series=getattr(agent, "reflection_result_series_dict", {}),
                window=self.requested_test_window,
            )
        if self.save_query_trace:
            self._write_agent_series(
                phase_name="test",
                output_name="test_query_trace.jsonl",
                series=getattr(agent, "query_trace_series_dict", {}),
                window=self.requested_test_window,
            )
        if self.save_llm_trace:
            self._write_agent_series(
                phase_name="test",
                output_name="test_llm_trace.jsonl",
                series=getattr(agent, "llm_trace_series_dict", {}),
                window=self.requested_test_window,
            )
