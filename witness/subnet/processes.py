"""Bounded subprocesses with process-group cancellation and output limits."""
from __future__ import annotations
import asyncio
import os
import signal


async def run_process(command: list[str], *, timeout: float, input_bytes: bytes = b"",
                      max_output: int = 2 * 1024 * 1024, env: dict | None = None,
                      new_session: bool = True) -> bytes:
    # Nested decoders inside an already isolated worker must inherit its group,
    # so the outer owner can terminate the complete task even after a hard kill.
    process = await asyncio.create_subprocess_exec(*command, stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL,
        start_new_session=new_session, env=env)
    async def read():
        result = bytearray()
        while chunk := await process.stdout.read(65536):
            result.extend(chunk)
            if len(result) > max_output:
                raise ValueError("worker_output_too_large")
        return bytes(result)
    async def write():
        try:
            process.stdin.write(input_bytes)
            await process.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            process.stdin.close()
    io_tasks = [asyncio.create_task(write()), asyncio.create_task(read())]
    try:
        async with asyncio.timeout(timeout):
            _, output = await asyncio.gather(*io_tasks)
            if await process.wait() != 0:
                raise RuntimeError("worker_failed")
            return output
    finally:
        for task in io_tasks:
            task.cancel()
        await asyncio.gather(*io_tasks, return_exceptions=True)
        # A cancelled/failed reader can leave stdout paused above its buffer
        # limit. Even a reaped child may then leave Process.wait() blocked.
        # Discard remaining output in bounded chunks throughout termination.
        async def drain():
            while await process.stdout.read(65536):
                pass
        draining = asyncio.create_task(drain())
        # Kill descendants even if the group leader exited early. No model,
        # decoder or nested worker survives cancellation of its task.
        try:
            if new_session:os.killpg(process.pid, signal.SIGTERM)
            elif process.returncode is None:process.terminate()
        except ProcessLookupError:
            pass
        try:
            await asyncio.wait_for(process.wait(), .5)
        except TimeoutError:
            pass
        try:
            if new_session:os.killpg(process.pid, signal.SIGKILL)
            elif process.returncode is None:process.kill()
        except ProcessLookupError:
            pass
        await asyncio.gather(process.wait(), draining)
