from build_manager import (
    BuildManager as bm,
)
import os
import shutil
import logging
import tarfile
from metadata_manager import (
    APSourceMetadataFetcher as apfetch,
    VehiclesManager as vehm
)
from shared_repo import SharedRepoManager
from pathlib import Path

CBS_BUILD_TIMEOUT_SEC = int(os.getenv('CBS_BUILD_TIMEOUT_SEC', 900))


class Builder:
    """
    Processes build requests using SharedRepoManager for efficient
    source provisioning (git worktrees) and build caching.
    """

    def __init__(self, workdir: str) -> None:
        if bm.get_singleton() is None:
            raise RuntimeError(
                "BuildManager should be initialized first."
            )
        if apfetch.get_singleton() is None:
            raise RuntimeError(
                "APSourceMetadataFetcher should be initialised first."
            )
        if vehm.get_singleton() is None:
            raise RuntimeError(
                "VehiclesManager should be initialised first."
            )
        if SharedRepoManager.get_singleton() is None:
            raise RuntimeError(
                "SharedRepoManager should be initialized first."
            )

        self.__workdir_parent = workdir
        self.__repo_mgr = SharedRepoManager.get_singleton()
        self.logger = logging.getLogger(__name__)
        self.__shutdown_requested = False

    # ------------------------------------------------------------------
    # Build processing pipeline
    # ------------------------------------------------------------------

    def __process_build(self, build_id: str) -> None:
        """Process a build: resolve, cache-or-build, archive, cleanup."""
        build_info = bm.get_singleton().get_build_info(build_id)
        vehicle = vehm.get_singleton().get_vehicle_by_id(
            build_info.vehicle_id
        )

        self.__create_build_workdir(build_id)
        self.__create_build_artifacts_dir(build_id)

        logpath = bm.get_singleton().get_build_log_path(build_id)
        with open(logpath, "a") as log_file:
            try:
                # 1. Log build info
                self.__write_build_info(build_info, log_file)

                # 2. Resolve commit SHA
                log_file.write(
                    f"=== Resolving {build_info.remote_info.name}/"
                    f"{build_info.git_hash} ===\n"
                )
                log_file.flush()
                self.__repo_mgr.ensure_remote(
                    build_info.remote_info.name,
                    build_info.remote_info.url,
                )
                commit_sha = self.__repo_mgr.resolve_commit(
                    build_info.remote_info.name,
                    build_info.git_hash,
                )
                log_file.write(f"Resolved to commit: {commit_sha[:12]}\n")
                log_file.flush()

                # 3. Generate extra_hwdef content
                hwdef_content = self.__generate_hwdef_content(
                    build_id, build_info, log_file
                )

                # 4. Build (or use cached build template)
                build_tpl = self.__repo_mgr.get_or_create_build_template(
                    commit=commit_sha,
                    board=build_info.board,
                    vehicle_waf_cmd=vehicle.waf_build_command,
                    extra_hwdef_content=hwdef_content,
                    log_file=log_file,
                    build_timeout=CBS_BUILD_TIMEOUT_SEC,
                )

                # 5. Copy build output to per-build workdir
                log_file.write("Collecting build artifacts...\n")
                log_file.flush()
                self.__collect_artifacts(
                    build_id, build_info, build_tpl
                )

                log_file.write("done build\n")
                log_file.flush()

            except Exception as e:
                self.logger.error(
                    "Build %s failed: %s", build_id, e, exc_info=True
                )
                log_file.write(f"\nBUILD FAILED: {e}\n")
                log_file.flush()

        # 6. Generate archive and cleanup
        self.__generate_archive(build_id)
        self.__clean_up_build_workdir(build_id)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def __write_build_info(self, build_info, log_file) -> None:
        """Write build metadata to the log."""
        log_file.write(
            f"Vehicle ID: {build_info.vehicle_id}\n"
            f"Board: {build_info.board}\n"
            f"Remote URL: {build_info.remote_info.url}\n"
            f"git-sha: {build_info.git_hash}\n"
            "---\n"
            "Selected Features:\n"
        )
        for d in build_info.selected_features:
            log_file.write(f"{d}\n")
        log_file.write("---\n")
        log_file.flush()

    def __generate_hwdef_content(self, build_id: str,
                                build_info, log_file) -> str:
        """
        Generate extra_hwdef.dat content as a string.
        Also writes it to the per-build workdir for archiving.
        """
        log_file.write("Generating extrahwdef...\n")
        log_file.flush()

        all_features = apfetch.get_singleton().get_build_options_at_commit(
            remote=build_info.remote_info.name,
            commit_ref=build_info.git_hash,
        )
        all_defines = {f.define for f in all_features}
        enabled_defines = build_info.selected_features.intersection(
            all_defines
        )
        disabled_defines = all_defines.difference(enabled_defines)

        lines = []
        # Undefine all first
        for define in sorted(all_defines):
            lines.append(f"undef {define}")
        # Enable selected
        for define in sorted(enabled_defines):
            lines.append(f"define {define} 1")
        # Disable the rest
        for define in sorted(disabled_defines):
            lines.append(f"define {define} 0")

        content = "\n".join(lines) + "\n"

        # Also write to per-build workdir for inclusion in archive
        hwdef_path = self.__get_path_to_extra_hwdef(build_id)
        os.makedirs(os.path.dirname(hwdef_path), exist_ok=True)
        with open(hwdef_path, "w") as f:
            f.write(content)

        return content

    def __collect_artifacts(self, build_id: str, build_info,
                           build_tpl: Path) -> None:
        """Copy compiled binaries from build template to workdir."""
        # waf puts output in build/<board>/bin/ within the build template
        bin_src = build_tpl / "build" / build_info.board / "bin"
        bin_dest = Path(self.__get_path_to_build_dir(build_id)) / \
            build_info.board / "bin"

        if bin_src.exists():
            bin_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(bin_src, bin_dest, dirs_exist_ok=True)
            self.logger.info("Copied binaries from %s", bin_src)
        else:
            self.logger.warning("No bin dir at %s", bin_src)
            bin_dest.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Directory management
    # ------------------------------------------------------------------

    def __create_build_artifacts_dir(self, build_id: str) -> None:
        p = Path(bm.get_singleton().get_build_artifacts_dir_path(build_id))
        self.logger.info(f"Creating directory at {p}.")
        try:
            Path.mkdir(p, parents=True)
        except FileExistsError:
            shutil.rmtree(p)
            Path.mkdir(p)

    def __create_build_workdir(self, build_id: str) -> None:
        p = Path(self.__get_path_to_build_dir(build_id))
        self.logger.info(f"Creating directory at {p}.")
        try:
            Path.mkdir(p, parents=True)
        except FileExistsError:
            shutil.rmtree(p)
            Path.mkdir(p)

    def __get_path_to_build_dir(self, build_id: str) -> str:
        return os.path.join(self.__workdir_parent, build_id)

    def __get_path_to_extra_hwdef(self, build_id: str) -> str:
        return os.path.join(
            self.__get_path_to_build_dir(build_id),
            "extra_hwdef.dat",
        )

    # ------------------------------------------------------------------
    # Archive generation
    # ------------------------------------------------------------------

    def __generate_archive(self, build_id: str) -> None:
        build_info = bm.get_singleton().get_build_info(build_id)
        archive_path = bm.get_singleton().get_build_archive_path(build_id)

        files_to_include = []

        # Binaries
        bin_path = os.path.join(
            self.__get_path_to_build_dir(build_id),
            build_info.board, "bin"
        )
        Path(bin_path).mkdir(parents=True, exist_ok=True)
        for file in os.listdir(bin_path):
            files_to_include.append(
                os.path.abspath(os.path.join(bin_path, file))
            )

        # Build log
        files_to_include.append(
            os.path.abspath(
                bm.get_singleton().get_build_log_path(build_id)
            )
        )

        # extra_hwdef.dat
        hwdef = self.__get_path_to_extra_hwdef(build_id)
        if os.path.exists(hwdef):
            files_to_include.append(os.path.abspath(hwdef))

        with tarfile.open(archive_path, "w:gz") as tar:
            for file in files_to_include:
                arcname = f"{build_id}/{os.path.basename(file)}"
                tar.add(file, arcname=arcname)
        self.logger.info(f"Generated {archive_path}.")

    def __clean_up_build_workdir(self, build_id: str) -> None:
        path = self.__get_path_to_build_dir(build_id)
        if os.path.exists(path):
            shutil.rmtree(path)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self.logger.info("Shutdown requested")
        self.__shutdown_requested = True

    def run(self) -> None:
        self.logger.info("Builder started and waiting for builds...")
        while not self.__shutdown_requested:
            build_to_process = bm.get_singleton().get_next_build_id(
                timeout=5
            )
            if build_to_process is None:
                continue

            self.logger.info(f"Processing build {build_to_process}")
            self.__process_build(build_id=build_to_process)

        self.logger.info("Builder shutting down gracefully")
