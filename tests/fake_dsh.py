"""Offline DSH transport double; it does not emulate the retired Python tool loop."""
from pathlib import Path

class FakeDSHHost:
    def __init__(self, *args, **kwargs):
        self.instance_dir = Path(args[0]) if args else Path("/unused-fake-host")
        self.sessions = {}
        self.messages = []
        self.started = False
        self.handler = None
    def set_callbacks(self, **kwargs):
        self.callbacks = kwargs
    async def start(self, **kwargs):
        self.started = True
    async def stop(self):
        self.started = False
    async def ensure_session(self, **kwargs):
        self.sessions[kwargs['session_id']] = kwargs
        return {}
    async def send_and_wait_quiet(self, **kwargs):
        self.messages.append(kwargs)
        if self.handler is not None:
            return await self.handler(kwargs)
        return {'finalText': 'DSH test response', 'status': 'idle'}
    async def send_message(self, **kwargs):
        self.messages.append(kwargs)
        return {}
    async def cancel_session(self, *args, **kwargs):
        return {}
    async def api(self, *args, **kwargs):
        return {}
    async def get_session_status(self, *args, **kwargs):
        return {'status': 'idle'}

    def collect_dsh_session_telemetry(self):
        return {}
    def _iter_decompressed_sessions(self):
        return []
