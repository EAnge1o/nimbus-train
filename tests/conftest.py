import os

# Keep the elementwise thread pool small under pytest: tests use tiny arrays
# where the serial path is faster anyway, and CI machines are shared.
os.environ.setdefault("NIMBUS_NUM_WORKERS", "2")
