"""Bounded owned child processes, including media preparation."""
import os
import signal
import subprocess
import time


def run_process(command, *, timeout, cancelled=lambda: False, remaining_s=lambda: float('inf'),
                input=None, check=False, capture_output=True, text=False, env=None):
    if cancelled() or remaining_s() <= 0:
        raise InterruptedError('evaluation_cancelled_or_budget_expired')
    until = time.monotonic() + min(timeout, remaining_s())
    process = subprocess.Popen(command, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=text,
                               env=env, start_new_session=True)
    try:
        while True:
            if cancelled() or time.monotonic() >= until:
                raise InterruptedError('evaluation_cancelled_or_budget_expired')
            try:
                out, err = process.communicate(input=input, timeout=min(.2, max(.001, until-time.monotonic())))
                result = subprocess.CompletedProcess(command, process.returncode, out, err)
                if check:
                    result.check_returncode()
                return result
            except subprocess.TimeoutExpired:
                input = None
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.communicate()
