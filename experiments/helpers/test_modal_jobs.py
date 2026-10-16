"""Dispatch/collection checks without submitting scientific work or contacting Modal."""

import json
from types import SimpleNamespace

import pytest
from experiments.helpers import modal_jobs as jobs
from pingstore.contracts import (
    PingstoreError,
    load_json,
    payload_digest,
    write_json_atomic,
)
from pingstore.layout import initialize_layout


class Worker:
    def __init__(self, root):
        self.root, self.calls = root, 0

    def with_options(self, **options):
        return self

    def spawn(self, experiment, reservation, dispatch, **arguments):
        self.calls += 1
        self.identity = reservation['run_id']
        receipt = load_json(self.root / 'runs' / f'.{self.identity}.tmp' / jobs.RECEIPT)
        assert receipt['function_call_id'] is None
        assert arguments['submission'] == receipt
        # No get() exists: dispatch must return without waiting for this call.
        return SimpleNamespace(object_id='fc-fixture')


def submission(root, inputs=None):
    worker = Worker(root)
    receipt = jobs.submit(root, 'exp999', worker, options={'gpu': None, 'timeout': 86400},
                          arguments={'smoke': True}, app_id='ap-fixture', inputs=inputs)
    assert worker.calls == 1
    return receipt


class Volume:
    def __init__(self, files):
        self.files, self.requests, self.interrupt = files, [], None

    def read_file(self, path):
        self.requests.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        if self.interrupt == path:
            self.interrupt = None
            yield self.files[path][:2]
            raise ConnectionError('transfer interrupted')
        yield self.files[path]


def completed_volume(tmp_path, receipt, inputs=None):
    source = tmp_path / 'remote' / receipt['run_id']
    initialize_layout(source, receipt['experiment'])
    (source / 'export' / 'result.json').write_text('{"value": 42}')
    record = {'schema': 'pingstore.run/v4', 'run_id': receipt['run_id'],
              'experiment': receipt['experiment'], 'stage': 'compute', 'scope': 'full',
              'collection': 'fixture', 'origin': f"modal-{receipt['dispatch']}",
              'created_at': '2026-10-15T10:00:00+00:00', 'inputs': inputs or {},
              'execution': {}, 'provenance': {}}
    record['payload_digest'] = payload_digest(source)
    write_json_atomic(source / 'run.json', record)
    files = {f"{receipt['remote_path']}/{p.relative_to(source)}": p.read_bytes()
             for p in source.rglob('*') if p.is_file()}
    names = [p.relative_to(source).as_posix() for p in source.rglob('*') if p.is_file()]
    files[f"{receipt['dispatch']}/completion.json"] = json.dumps(
        {'run_id': receipt['run_id'], 'dispatch': receipt['dispatch'], 'files': names}).encode()
    return Volume(files)


def no_call(_):
    raise AssertionError('durable volume completion must work without a live/retained call')


def test_dispatch_returns_with_saved_call_id(tmp_path):
    receipt = submission(tmp_path)
    assert load_json(tmp_path / 'runs' / f".{receipt['run_id']}.tmp" / jobs.RECEIPT) == receipt


def test_execution_plan_is_persisted_before_spawn(tmp_path):
    plan = {"configuration": {"models": ["coba", "ping"]}, "resources": {"gpu": "A100-80GB:2"}}
    worker = Worker(tmp_path)
    receipt = jobs.submit(tmp_path, 'exp135', worker, options=plan['resources'],
                          arguments={}, app_id='ap-fixture', plan=plan)
    assert worker.calls == 1
    assert receipt['plan'] == plan
    assert jobs.pending(tmp_path, receipt['run_id'])[2]['plan'] == plan


def test_failed_spawn_keeps_prepared_receipt_and_never_retries(tmp_path):
    worker = Worker(tmp_path)
    def failed(*args, **kwargs):
        worker.calls += 1
        raise ConnectionError('submission acknowledgement lost')
    worker.spawn = failed
    with pytest.raises(ConnectionError):
        jobs.submit(tmp_path, 'exp999', worker, options={}, arguments={}, app_id='ap-fixture')
    assert worker.calls == 1
    receipt = load_json(next((tmp_path / 'runs').glob('.*.tmp')) / jobs.RECEIPT)
    assert receipt['function_call_id'] is None
    _, state = jobs.completed_files(receipt, Volume({}), no_call)
    assert state == 'submission_unknown'


def test_pending_poll_is_nonblocking_and_does_not_download(tmp_path):
    receipt = submission(tmp_path)
    def factory(identity):
        assert identity == 'fc-fixture'
        def get(*, timeout):
            assert timeout == 0
            raise TimeoutError()
        return SimpleNamespace(get=get)
    result = jobs.check(tmp_path, receipt['run_id'], collect=True, volume=Volume({}), call_factory=factory)
    assert result['status'] == 'pending'
    assert not (tmp_path / 'runs' / receipt['run_id']).exists()


def test_remote_failure_propagates_without_resubmission(tmp_path):
    receipt = submission(tmp_path)
    def get(**_):raise RuntimeError('remote failed')
    with pytest.raises(RuntimeError, match='remote failed'):
        jobs.check(tmp_path, receipt['run_id'], volume=Volume({}),
                   call_factory=lambda _: SimpleNamespace(get=get))


def test_collect_after_dispatcher_exits_and_repeat_without_network(tmp_path):
    receipt = submission(tmp_path)
    volume = completed_volume(tmp_path, receipt)
    result = jobs.check(tmp_path, receipt['run_id'], collect=True, volume=volume, call_factory=no_call)
    assert result['status'] == 'collected'
    destination = tmp_path / 'runs' / receipt['run_id']
    record = load_json(destination / 'run.json')
    assert record['execution']['modal']['function_call_id'] == 'fc-fixture'
    assert not (tmp_path / 'runs' / f".{receipt['run_id']}.tmp").exists()
    assert jobs.check(tmp_path, receipt['run_id'])['status'] == 'collected'


def test_interrupted_transfer_retries_without_spawning(tmp_path):
    receipt = submission(tmp_path)
    volume = completed_volume(tmp_path, receipt)
    volume.interrupt = f"{receipt['remote_path']}/export/result.json"
    with pytest.raises(ConnectionError):
        jobs.check(tmp_path, receipt['run_id'], collect=True, volume=volume, call_factory=no_call)
    assert not (tmp_path / 'runs' / receipt['run_id']).exists()
    assert jobs.check(tmp_path, receipt['run_id'], collect=True, volume=volume,
                      call_factory=no_call)['status'] == 'collected'


@pytest.mark.parametrize('corruption', ['path', 'origin', 'digest'])
def test_invalid_download_never_becomes_visible(tmp_path, corruption):
    receipt = submission(tmp_path)
    volume = completed_volume(tmp_path, receipt)
    key = f"{receipt['dispatch']}/completion.json"
    if corruption == 'path':
        marker = json.loads(volume.files[key])
        marker['files'].append('../../escape')
        volume.files[key] = json.dumps(marker).encode()
    elif corruption == 'origin':
        key = f"{receipt['remote_path']}/run.json"
        record = json.loads(volume.files[key])
        record['origin'] = 'wrong'
        volume.files[key] = json.dumps(record).encode()
    else:
        volume.files[f"{receipt['remote_path']}/export/result.json"] = b'bad digest'
    with pytest.raises(PingstoreError):
        jobs.check(tmp_path, receipt['run_id'], collect=True, volume=volume, call_factory=no_call)
    assert not (tmp_path / 'runs' / receipt['run_id']).exists()


def test_collect_phase_preserves_scope(tmp_path):
    receipt = submission(tmp_path)
    identity = jobs.reserve_stage(tmp_path, 'exp999', 'compute',
                                  origin=f"modal-{receipt['dispatch']}", phase='gpu-benchmark')
    receipt = {**receipt, 'run_id': identity,
               'remote_path': f"{receipt['dispatch']}/.pingstore/runs/{identity}"}
    write_json_atomic(tmp_path / 'runs' / f'.{identity}.tmp' / jobs.RECEIPT, receipt)
    volume = completed_volume(tmp_path, receipt)
    key = f"{receipt['remote_path']}/run.json"
    record = json.loads(volume.files[key])
    record.update(scope='phase', phase='gpu-benchmark')
    volume.files[key] = json.dumps(record).encode()
    result = jobs.check(tmp_path, identity, collect=True, volume=volume, call_factory=no_call)
    assert result['status'] == 'collected'
    assert load_json(tmp_path / 'runs' / identity / 'run.json')['phase'] == 'gpu-benchmark'
