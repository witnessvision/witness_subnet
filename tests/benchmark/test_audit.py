import json
from witness.benchmark.audit import record, _handlers


def test_audit_is_bounded_private_and_redacts_identity(tmp_path,monkeypatch):
    p=tmp_path/'access.jsonl';monkeypatch.setenv('WITNESS_AUDIT_LOG',str(p))
    record('model_access',status=403,peer='bad\ninput',identity='secret')
    row=json.loads(p.read_text());assert row['peer_ip'] is None and row['authenticated_hotkey'] is None
    handler=_handlers[str(p)];handler.maxBytes=1
    record('model_access',status=200)
    assert len(list(tmp_path.iterdir()))==2
    assert all(f.stat().st_mode&0o777==0o600 for f in tmp_path.iterdir())
