import logging

from tmu.tools import BenchmarkTimer

_LOGGER = logging.getLogger(__name__)

try:
    import pycuda._driver
    import pycuda.autoinit
    import pycuda.driver as drv
    import os


    class CudaProfiler:
        _instance = None

        def __new__(cls, *args, **kwargs):
            if not cls._instance:
                cls._instance = super(CudaProfiler, cls).__new__(cls, *args, **kwargs)
            return cls._instance

        def __init__(self):
            if not hasattr(self, 'timings'):
                self.timings = []
                if os.getenv("CUDA_PROFILE"):
                    self.fn = self.profile_fn
                else:
                    self.fn = self.profile_dummy

        def profile(self, operation, *args):
            return self.fn(operation, *args)

        def profile_dummy(self, operation, *args):
            return operation(*args)

        def profile_fn(self, operation, *args):
            start_event = drv.Event()
            end_event = drv.Event()

            start_event.record()
            ret = operation(*args)
            end_event.record()

            end_event.synchronize()

            self.timings.append({
                "time": start_event.time_till(end_event) / 1000.0,
                "op": operation.__name__,
            })

            return ret

        def print_timings(self, benchmark: BenchmarkTimer):
            total_time = benchmark.elapsed()

            # Sum different ops
            timings = {}
            for timing in self.timings:
                if timing["op"] not in timings:
                    timings[timing["op"]] = 0
                timings[timing["op"]] += timing["time"]

            # Total time
            total_gpu_time = sum(timing["time"] for timing in self.timings)

            # Print gpu time, time per op and percentage of total time
            _LOGGER.info(f"Total GPU time: {total_gpu_time:.2f}s")
            _LOGGER.info(f"Total time: {total_time:.2f}s")
            _LOGGER.info(f"Percentage of total time: {total_gpu_time / total_time * 100:.2f}%")
            for op, time in timings.items():
                _LOGGER.info(f"{op}: {time:.2f}s, {time / total_time * 100:.2f}%")

            # Clear timings
            self.timings.clear()


except Exception as e:
    _LOGGER.warning(f"Could not import pycuda: {e}")


    class CudaProfiler:

        _instance = None

        def __new__(cls, *args, **kwargs):
            if not cls._instance:
                cls._instance = super(CudaProfiler, cls).__new__(cls, *args, **kwargs)
            return cls._instance

        def __init__(self):
            pass

        def profile(self, operation, *args):
            return
