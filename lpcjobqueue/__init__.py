import os
import re
import logging
import asyncio
import weakref
import socket
from dask_jobqueue.htcondor import (
    HTCondorCluster,
    HTCondorJob,
    quote_arguments,
    quote_environment,
)
import dask
from distributed.core import Status
import yaml

fn = os.path.join(os.path.dirname(__file__), "config.yaml")
dask.config.ensure_file(source=fn)

with open(fn) as f:
    defaults = yaml.safe_load(f)

dask.config.update(dask.config.config, defaults, priority="old")

os.environ["CONDOR_CONFIG"] = os.path.join(os.path.dirname(__file__), "condor_config")
import htcondor  # noqa: E402

logger = logging.getLogger(__name__)


def acquire_schedd():
    """Acquire a htcondor.Schedd object

    Uses the bundled condor_config to connect to the LPC pool, query available schedds,
    and use the custom `condor_submit` schedd-choosing algorithm to select a schedd for
    this session. This function will not return the same value, so keep it around until
    all jobs are removed!
    """
    remotePool = re.findall(
        r"[\w\/\:\/\-\/\.]+", htcondor.param.get("FERMIHTC_REMOTE_POOL")
    )
    collector = None
    scheddAds = None
    for node in remotePool:
        try:
            collector = htcondor.Collector(node)
            scheddAds = collector.query(
                htcondor.AdTypes.Schedd,
                projection=[
                    "Name",
                    "MyAddress",
                    "MaxJobsRunning",
                    "ShadowsRunning",
                    "RecentDaemonCoreDutyCycle",
                    "TotalIdleJobs",
                ],
                constraint='FERMIHTC_DRAIN_LPCSCHEDD=?=FALSE && FERMIHTC_SCHEDD_TYPE=?="CMSLPC"',
            )
            break
        except Exception:
            logger.debug(f"Failed to contact pool node {node}, trying others...")
            pass

    if not scheddAds:
        raise RuntimeError("No pool nodes could be contacted")

    weightedSchedds = {}
    for schedd in scheddAds:
        # covert duty cycle in percentage
        scheddDC = schedd["RecentDaemonCoreDutyCycle"] * 100
        # calculate schedd occupancy in terms of running jobs
        scheddRunningJobs = (schedd["ShadowsRunning"] / schedd["MaxJobsRunning"]) * 100

        logger.debug("Looking at schedd: " + schedd["Name"])
        logger.debug(f"DutyCyle: {scheddDC}%")
        logger.debug(f"Running percentage: {scheddRunningJobs}%")
        logger.debug(f"Idle jobs: {schedd['TotalIdleJobs']}")

        # Calculating weight
        # 70% of schedd duty cycle
        # 20% of schedd capacity to run more jobs
        # 10% of idle jobs on the schedd (for better distribution of jobs across all schedds)
        weightedSchedds[schedd["Name"]] = (
            (0.7 * scheddDC)
            + (0.2 * scheddRunningJobs)
            + (0.1 * schedd["TotalIdleJobs"])
        )

    schedd = min(weightedSchedds.items(), key=lambda x: x[1])[0]
    schedd = collector.locate(htcondor.DaemonTypes.Schedd, schedd)
    return htcondor.Schedd(schedd)


# Pick a schedd once on import
# Would prefer one per cluster but there is a quite scary weakref.finalize
# that depends on it
SCHEDD = acquire_schedd()


class LPCCondorJob(HTCondorJob):
    executable = "/usr/bin/env"
    config_name = "lpccondor"
    known_jobs = set()

    def __init__(self, scheduler=None, name=None, **base_class_kwargs):
        base_class_kwargs["python"] = "python"
        super().__init__(scheduler=scheduler, name=name, **base_class_kwargs)
        homedir = os.path.expanduser("~")
        if self.log_directory:
            if not self.log_directory.startswith(homedir):
                raise ValueError(
                    f"log_directory must be a subpath of {homedir} or else the schedd cannot write our logs back to the container"
                )
            self.job_header_dict.pop("Stream_Output")
            self.job_header_dict.pop("Stream_Error")

        self.job_header_dict.update(
            {
                "initialdir": homedir,
                "use_x509userproxy": "true",
                "when_to_transfer_output": "ON_EXIT_OR_EVICT",
                "+SingularityImage": '"/cvmfs/unpacked.cern.ch/registry.hub.docker.com/coffeateam/coffea-dask:latest"',
            }
        )

    def job_script(self):
        """ Construct a job submission script """
        quoted_arguments = quote_arguments(self._command_template.split(" "))
        quoted_environment = quote_environment(self.env_dict)
        job_header_lines = "\n".join(
            "%s = %s" % (k, v) for k, v in self.job_header_dict.items()
        )
        return self._script_template % {
            "shebang": self.shebang,
            "job_header": job_header_lines,
            "quoted_environment": quoted_environment,
            "quoted_arguments": quoted_arguments,
            "executable": self.executable,
        }

    async def start(self):
        """ Start workers and point them to our local scheduler """
        logger.debug("Starting worker: %s", self.name)

        job = self.job_script()
        logger.debug(job)
        job = htcondor.Submit(job)

        def sub():
            try:
                classads = []
                with SCHEDD.transaction() as txn:
                    cluster_id = job.queue(txn, ad_results=classads)

                logger.debug(classads)
                SCHEDD.spool(classads)
                return cluster_id
            except htcondor.HTCondorInternalError as ex:
                logger.error(str(ex))
                return None

        self.job_id = await asyncio.get_event_loop().run_in_executor(None, sub)
        if self.job_id:
            self.known_jobs.add(self.job_id)
            weakref.finalize(self, self._close_job, self.job_id)

            logger.debug("Starting job: %s", self.job_id)
            # await super().start() all this does is set a flag
            self.status = Status.running

    async def close(self):
        logger.debug("Forcefully stopping worker: %s job: %s", self.name, self.job_id)

        def stop():
            return SCHEDD.act(htcondor.JobAction.Remove, f"ClusterId == {self.job_id}")

        result = await asyncio.get_event_loop().run_in_executor(None, stop)
        logger.debug(f"Closed job {self.job_id}, result {result}")
        self.known_jobs.remove(self.job_id)

    @classmethod
    def _close_job(cls, job_id):
        if job_id in cls.known_jobs:
            logger.info(f"Closeing job {job_id} in a finalizer")
            result = SCHEDD.act(htcondor.JobAction.Remove, f"ClusterId == {job_id}")
            logger.debug(f"Closed job {job_id}, result {result}")
            cls.known_jobs.remove(job_id)


class LPCCondorCluster(HTCondorCluster):
    __doc__ = (
        HTCondorCluster.__doc__
        + """
        More LPC-specific info...
    """
    )
    job_cls = LPCCondorJob
    config_name = "lpccondor"

    def __init__(self, **kwargs):
        hostname = socket.gethostname()
        port = 10000
        scheduler_options = {"host": f"{hostname}:{port}"}
        if "scheduler_options" in kwargs:
            kwargs["scheduler_options"].update(scheduler_options)
        else:
            kwargs["scheduler_options"] = scheduler_options
        super().__init__(**kwargs)
