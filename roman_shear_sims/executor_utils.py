import concurrent.futures
import atexit
from joblib.externals.loky import get_reusable_executor

_executor = None
# _executor = get_reusable_executor()


def get_executor():
    global _executor
    if _executor is None:
        # _executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        _executor = get_reusable_executor(max_workers=1)
    return _executor


def cleanup_executor():
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None


atexit.register(cleanup_executor)
