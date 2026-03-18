import os
import sys
import signal
import logging
from ap_git import GitRepo
from build_manager import BuildManager
from builder import Builder
from shared_repo import SharedRepoManager
from metadata_manager import (
    APSourceMetadataFetcher,
    VehiclesManager,
)
from logging.config import dictConfig

dictConfig({
    'version': 1,
    'formatters': {
        'default': {
            'format': '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        },
    },
    'handlers': {
        'stream': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'default',
            'stream': sys.stdout,
        },
    },
    'loggers': {
        'root': {
            'level': 'INFO',
            'handlers': ['stream'],
        },
    },
})

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    basedir = os.path.abspath(os.getenv("CBS_BASEDIR"))
    workdir = os.path.abspath('/workdir')

    # Shared golden repo (used by builder for worktrees + build caching)
    shared_repo_dir = os.environ.get(
        "SHARED_ARDUPILOT_DIR",
        os.path.join(workdir, "shared-ardupilot"),
    )
    shared_repo_mgr = SharedRepoManager(
        repo_path=shared_repo_dir,
        clone_url="https://github.com/ardupilot/ardupilot.git",
    )

    # Metadata repo (separate clone for APSourceMetadataFetcher —
    # it checks out commits to read build options from source tree)
    metadata_repo = GitRepo.clone_if_needed(
        source="https://github.com/ardupilot/ardupilot.git",
        dest=os.path.join(basedir, 'ardupilot'),
    )

    ap_metafetch = APSourceMetadataFetcher(
        ap_repo=metadata_repo
    )

    vehicles_manager = VehiclesManager()

    manager = BuildManager(
        outdir=os.path.join(basedir, 'artifacts'),
        redis_host=os.getenv('CBS_REDIS_HOST', default='localhost'),
        redis_port=os.getenv('CBS_REDIS_PORT', default='6379')
    )

    builder = Builder(
        workdir=workdir,
    )

    # Set up signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        signame = signal.Signals(signum).name
        logger.info(f"Received {signame}, initiating graceful shutdown...")
        builder.shutdown()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info("Builder starting...")
    builder.run()
    logger.info("Builder exited gracefully")
