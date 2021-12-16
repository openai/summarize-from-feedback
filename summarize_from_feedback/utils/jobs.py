import os
import subprocess
from dataclasses import dataclass, field
from typing import Callable, List

import cloudpickle

from summarize_from_feedback.utils import hyperparams


@dataclass
class JobHParams(hyperparams.HParams):
    name: str = None
    mode: str = "local"
    mpi: int = 1


@dataclass
class Job(hyperparams.HParams):
    fn: Callable = None
    params: JobHParams = field(default_factory=JobHParams)


def launch(job: Job):
    H = job.params
    if H.mode == "local":
        with open("/tmp/pickle_fn", "wb") as file:
            cloudpickle.dump(job.fn, file)

        env = os.environ.copy()
        env["JOB_NAME"] = H.name

        subprocess.check_call(
            [
                #"mpiexec",
                #"-n",
                #str(H.mpi),
                "python",
                "-c",
                'import sys; import pickle; pickle.loads(open("/tmp/pickle_fn", "rb").read())()',
            ],
            env=env,
        )
    else:
        raise NotImplementedError(f"unsupported launch mode {H.mode}")


def multilaunch(jobs: List[Job]):
    return [launch(job) for job in jobs]
