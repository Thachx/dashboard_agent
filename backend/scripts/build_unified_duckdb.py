#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_DATA_DIR = Path('/home/thacha/dashboard_agent/data')
DEFAULT_OUT_DIR = DEFAULT_DATA_DIR / '_warehouse'
SUPPORTED_SUFFIXES = {'.json', '.csv', '.parquet', '.sql'}
SKIP_DIR_NAMES = {'_warehouse'}

SCHEMA = pa.schema([
    ('source_path', pa.string()),
    ('source_format', pa.string()),
    ('source_table', pa.string()),
    ('record_index', pa.int64()),
    ('payload_json', pa.large_string()),
    ('ingested_at', pa.timestamp('us', tz='UTC')),
])


def log(message: str) -> None:
    now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    print(f'{now} {message}', flush=True)


def table_name_from_path(path: Path, data_dir: Path) -> str:
    rel = path.relative_to(data_dir).as_posix()
    rel = re.sub(r'\.[^.]+$', '', rel)
    return re.sub(r'[^A-Za-z0-9_]+', '_', rel).strip('_') or 'root'


def iter_input_files(data_dir: Path, include_sql: bool) -> Iterator[Path]:
    for path in sorted(data_dir.rglob('*')):
        if not path.is_file():
            continue
        if any(part in SKIP_DIR_NAMES for part in path.relative_to(data_dir).parts):
            continue
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            continue
        if suffix == '.sql' and not include_sql:
            continue
        yield path


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[object]:
    decoder = json.JSONDecoder()
    with path.open('r', encoding='utf-8', errors='replace') as handle:
        buffer = ''
        eof = False
        started = False
        while True:
            if not eof:
                chunk = handle.read(chunk_size)
                if chunk:
                    buffer += chunk
                else:
                    eof = True

            while True:
                buffer = buffer.lstrip()
                if not started:
                    if not buffer:
                        break
                    if buffer[0] != '[':
                        raise ValueError('JSON document is not an array')
                    buffer = buffer[1:]
                    started = True
                    continue

                buffer = buffer.lstrip()
                if not buffer:
                    break
                if buffer[0] == ',':
                    buffer = buffer[1:]
                    continue
                if buffer[0] == ']':
                    return

                try:
                    value, idx = decoder.raw_decode(buffer)
                except json.JSONDecodeError:
                    if eof:
                        raise
                    break
                yield value
                buffer = buffer[idx:]

            if eof:
                if buffer.strip() not in {'', ']'}:
                    raise ValueError('unexpected trailing JSON content')
                return


def iter_json_records(path: Path) -> Iterator[str]:
    with path.open('r', encoding='utf-8', errors='replace') as handle:
        first = handle.read(1)
    if first == '[':
        for value in iter_json_array(path):
            yield json.dumps(value, ensure_ascii=False, separators=(',', ':'))
        return

    with path.open('r', encoding='utf-8', errors='replace') as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
                yield json.dumps(value, ensure_ascii=False, separators=(',', ':'))
            except json.JSONDecodeError:
                yield json.dumps({'line': text}, ensure_ascii=False, separators=(',', ':'))


def iter_csv_records(path: Path) -> Iterator[str]:
    with path.open('r', encoding='utf-8-sig', errors='replace', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield json.dumps(row, ensure_ascii=False, separators=(',', ':'))


def iter_sql_records(path: Path) -> Iterator[str]:
    with path.open('r', encoding='utf-8', errors='replace') as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.rstrip('\n')
            if not text:
                continue
            yield json.dumps({'line_number': line_number, 'sql': text}, ensure_ascii=False, separators=(',', ':'))


def iter_parquet_records(path: Path, batch_size: int) -> Iterator[str]:
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            yield json.dumps(row, ensure_ascii=False, default=str, separators=(',', ':'))


def iter_payloads(path: Path, batch_size: int) -> Iterator[str]:
    suffix = path.suffix.lower()
    if suffix == '.json':
        yield from iter_json_records(path)
    elif suffix == '.csv':
        yield from iter_csv_records(path)
    elif suffix == '.parquet':
        yield from iter_parquet_records(path, batch_size=batch_size)
    elif suffix == '.sql':
        yield from iter_sql_records(path)
    else:
        raise ValueError(f'unsupported file suffix: {path.suffix}')


def write_batch(writer: pq.ParquetWriter, rows: list[dict]) -> None:
    if not rows:
        return
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    writer.write_table(table)


def build_unified_table(
    data_dir: Path,
    out_dir: Path,
    include_sql: bool,
    batch_size: int,
    max_files: int | None,
    max_records_per_file: int | None,
) -> tuple[Path, Path, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / 'unified_records.parquet'
    duckdb_path = out_dir / 'dashboard_agent.duckdb'
    tmp_parquet_path = out_dir / 'unified_records.parquet.tmp'
    tmp_duckdb_path = out_dir / 'dashboard_agent.duckdb.tmp'

    for path in [tmp_parquet_path, tmp_duckdb_path]:
        if path.exists():
            path.unlink()

    files = list(iter_input_files(data_dir, include_sql=include_sql))
    if max_files is not None:
        files = files[:max_files]

    log(f'input files: {len(files)}')
    total_records = 0
    processed_files = 0
    ingested_at = datetime.now(timezone.utc)

    with pq.ParquetWriter(tmp_parquet_path, SCHEMA, compression='zstd') as writer:
        for path in files:
            rel = path.relative_to(data_dir).as_posix()
            source_format = path.suffix.lower().lstrip('.')
            source_table = table_name_from_path(path, data_dir)
            log(f'start {rel}')
            rows: list[dict] = []
            file_records = 0
            try:
                for payload in iter_payloads(path, batch_size=batch_size):
                    rows.append({
                        'source_path': rel,
                        'source_format': source_format,
                        'source_table': source_table,
                        'record_index': file_records,
                        'payload_json': payload,
                        'ingested_at': ingested_at,
                    })
                    file_records += 1
                    total_records += 1
                    if len(rows) >= batch_size:
                        write_batch(writer, rows)
                        rows.clear()
                    if max_records_per_file is not None and file_records >= max_records_per_file:
                        break
                write_batch(writer, rows)
                processed_files += 1
                log(f'done {rel}: {file_records} records')
            except Exception as exc:
                log(f'error {rel}: {type(exc).__name__}: {exc}')
                raise

    if tmp_duckdb_path.exists():
        tmp_duckdb_path.unlink()
    con = duckdb.connect(str(tmp_duckdb_path))
    try:
        con.execute('CREATE TABLE unified_records AS SELECT * FROM read_parquet(?)', [str(tmp_parquet_path)])
        con.execute('CREATE TABLE source_summary AS SELECT source_path, source_format, source_table, count(*) AS records FROM unified_records GROUP BY 1,2,3 ORDER BY records DESC')
        con.execute('CHECKPOINT')
    finally:
        con.close()

    tmp_parquet_path.replace(parquet_path)
    tmp_duckdb_path.replace(duckdb_path)
    log(f'complete: {processed_files} files, {total_records} records')
    return parquet_path, duckdb_path, processed_files, total_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Combine project data files into one generic Parquet table and DuckDB database.')
    parser.add_argument('--data-dir', type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument('--include-sql', action='store_true', help='Include .sql dump files as one record per non-empty line. This can be very large.')
    parser.add_argument('--batch-size', type=int, default=10000)
    parser.add_argument('--max-files', type=int, default=None, help='Debug limit.')
    parser.add_argument('--max-records-per-file', type=int, default=None, help='Debug limit.')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_unified_table(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        include_sql=args.include_sql,
        batch_size=args.batch_size,
        max_files=args.max_files,
        max_records_per_file=args.max_records_per_file,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
