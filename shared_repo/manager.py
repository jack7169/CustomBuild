"""
SharedRepoManager — shared golden ArduPilot repo with worktree-based
source templates and pre-built build template caching.

Ported from the autotest system's optimized approach, adapted for
the synchronous (threaded) custombuild builder.
"""
import fcntl
import hashlib
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class SharedRepoManager:
    """
    Manages a single shared ArduPilot git repository ("golden copy"),
    git worktree-based source templates, and pre-built build template
    caches.  Thread-safe via fine-grained locks; cross-process safe
    for git-fetch via file lock.
    """

    _singleton = None

    def __init__(self, repo_path: str, clone_url: str,
                 max_source_templates: int = 10,
                 max_build_templates: int = 20):
        if SharedRepoManager._singleton is not None:
            raise RuntimeError("SharedRepoManager must be a singleton.")

        self._repo = Path(repo_path)
        self._clone_url = clone_url
        self._templates_dir = self._repo.parent / "shared-templates"
        self._build_templates_dir = self._repo.parent / "shared-build-templates"
        self._max_source_tpl = max_source_templates
        self._max_build_tpl = max_build_templates

        # --- Locks (in-process, threading) ---
        self._fetch_lock = threading.Lock()
        self._tpl_cache_lock = threading.Lock()
        self._tpl_locks: dict[str, threading.Lock] = {}
        self._build_cache_lock = threading.Lock()
        self._build_key_locks: dict[str, threading.Lock] = {}

        # --- In-memory caches ---
        # source templates: commit_sha -> {"path": Path, "last_used": float}
        self._tpl_cache: dict[str, dict] = {}
        # build templates:  cache_key  -> {"path": Path, "last_used": float}
        self._build_cache: dict[str, dict] = {}

        # Bootstrap
        self._ensure_repo()
        self._templates_dir.mkdir(parents=True, exist_ok=True)
        self._build_templates_dir.mkdir(parents=True, exist_ok=True)
        self._restore_caches()

        SharedRepoManager._singleton = self
        logger.info("SharedRepoManager initialized at %s", self._repo)

    @staticmethod
    def get_singleton() -> "SharedRepoManager | None":
        return SharedRepoManager._singleton

    # ------------------------------------------------------------------
    # Repository bootstrap
    # ------------------------------------------------------------------

    def _ensure_repo(self):
        """Clone the golden repo if it doesn't exist yet."""
        waf = self._repo / "waf"
        if waf.exists():
            logger.info("Golden repo already exists at %s", self._repo)
            return

        # Clean incomplete clone
        if self._repo.exists():
            logger.warning("Removing incomplete clone at %s", self._repo)
            shutil.rmtree(self._repo, ignore_errors=True)

        self._repo.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Cloning golden repo from %s ...", self._clone_url)
        self._run(
            ["git", "clone", "--progress", "--recurse-submodules",
             self._clone_url, str(self._repo)],
            timeout=900,
        )
        logger.info("Golden repo cloned.")

        # Ensure submodules initialized
        if not (self._repo / "modules" / "mavlink" / ".git").exists():
            cpu = os.cpu_count() or 4
            self._run(
                ["git", "submodule", "update", "--init", "--recursive",
                 "--depth", "1", f"--jobs={cpu}"],
                cwd=self._repo, timeout=600,
            )

    def _restore_caches(self):
        """Reload template caches from disk on startup, prune stale."""
        # Build templates
        for item in self._build_templates_dir.iterdir():
            if item.is_dir() and item.name.startswith("bld-"):
                parts = item.name.split("-")
                key = parts[1] if len(parts) >= 3 else item.name
                self._build_cache[key] = {
                    "path": item, "last_used": time.time()
                }
        self._evict_build_templates()
        logger.info("Restored %d build template(s)", len(self._build_cache))

        # Source templates
        for tpl in self._templates_dir.iterdir():
            if tpl.is_dir() and tpl.name.startswith("tpl-"):
                prefix = tpl.name.removeprefix("tpl-")
                rc, out = self._run(
                    ["git", "rev-parse", prefix],
                    cwd=self._repo, timeout=10, check=False,
                )
                commit = out.strip() if rc == 0 else prefix
                self._tpl_cache[commit] = {
                    "path": tpl, "last_used": time.time()
                }
        self._evict_source_templates()
        self._run(["git", "worktree", "prune"],
                  cwd=self._repo, timeout=30, check=False)
        logger.info("Restored %d source template(s)", len(self._tpl_cache))

    # ------------------------------------------------------------------
    # Remote / fetch management
    # ------------------------------------------------------------------

    def ensure_remote(self, name: str, url: str):
        """Add or update a remote on the golden repo."""
        rc, out = self._run(
            ["git", "remote", "get-url", name],
            cwd=self._repo, check=False,
        )
        if rc != 0:
            self._run(["git", "remote", "add", name, url], cwd=self._repo)
            logger.info("Added remote %s: %s", name, url)
        elif out.strip() != url:
            self._run(
                ["git", "remote", "set-url", name, url], cwd=self._repo
            )
            logger.info("Updated remote %s URL to %s", name, url)

    def fetch_remote(self, name: str, ref: str | None = None):
        """Fetch a remote, serialized via in-process + file lock."""
        with self._fetch_lock:
            lock_path = self._repo / ".fetch.lock"
            with open(lock_path, "w") as lf:
                fcntl.flock(lf, fcntl.LOCK_EX)
                try:
                    cmd = ["git", "fetch", name, "--prune", "--tags",
                           "--no-recurse-submodules"]
                    if ref and len(ref) < 40:
                        cmd.append(
                            f"+refs/heads/{ref}:refs/remotes/{name}/{ref}"
                        )
                    self._run(cmd, cwd=self._repo, timeout=300)
                finally:
                    fcntl.flock(lf, fcntl.LOCK_UN)

    # ------------------------------------------------------------------
    # Commit resolution
    # ------------------------------------------------------------------

    def resolve_commit(self, remote: str, commit_ref: str) -> str:
        """
        Resolve a ref (branch name, tag, or SHA) to a full commit SHA.
        Fetches from remote if not found locally.
        """
        sha = self._try_resolve_local(remote, commit_ref)
        if sha:
            return sha

        # Not local — fetch and retry
        logger.info("Ref %s not local, fetching %s...", commit_ref, remote)
        self.fetch_remote(remote, ref=commit_ref)

        sha = self._try_resolve_local(remote, commit_ref)
        if sha:
            return sha

        raise ValueError(
            f"Could not resolve '{commit_ref}' on remote '{remote}' "
            "after fetching."
        )

    def _try_resolve_local(self, remote: str,
                           commit_ref: str) -> str | None:
        """Try to resolve a ref locally without network access."""
        # Already a full SHA?
        if len(commit_ref) == 40:
            rc, _ = self._run(
                ["git", "cat-file", "-t", commit_ref],
                cwd=self._repo, check=False,
            )
            if rc == 0:
                return commit_ref

        # Try remote/ref (branch)
        if len(commit_ref) < 40:
            rc, out = self._run(
                ["git", "rev-parse", "--verify",
                 f"{remote}/{commit_ref}"],
                cwd=self._repo, check=False,
            )
            if rc == 0:
                return out.strip()

        # Try as tag or bare ref
        rc, out = self._run(
            ["git", "rev-parse", "--verify",
             f"{commit_ref}^{{commit}}"],
            cwd=self._repo, check=False,
        )
        if rc == 0:
            return out.strip()

        return None

    # ------------------------------------------------------------------
    # Source template cache  (git worktrees with submodules)
    # ------------------------------------------------------------------

    def get_or_create_source_template(self, commit: str,
                                      log_file=None) -> Path:
        """
        Get a cached source template for a commit, or create one.
        Thread-safe with per-commit locking.
        """
        # Fast check
        with self._tpl_cache_lock:
            if commit in self._tpl_cache:
                entry = self._tpl_cache[commit]
                if entry["path"].exists():
                    entry["last_used"] = time.time()
                    self._log(log_file,
                              f"Source template cache HIT: {commit[:12]}\n")
                    return entry["path"]
                del self._tpl_cache[commit]

        # Per-commit lock
        with self._tpl_cache_lock:
            if commit not in self._tpl_locks:
                self._tpl_locks[commit] = threading.Lock()
            lock = self._tpl_locks[commit]

        with lock:
            # Double-check after acquiring lock
            with self._tpl_cache_lock:
                if commit in self._tpl_cache:
                    entry = self._tpl_cache[commit]
                    if entry["path"].exists():
                        entry["last_used"] = time.time()
                        return entry["path"]
                    del self._tpl_cache[commit]

            return self._create_source_template(commit, log_file)

    def _create_source_template(self, commit: str,
                                log_file=None) -> Path:
        """Create a new source template worktree."""
        self._log(log_file,
                  f"Source template cache MISS: {commit[:12]}, creating...\n")

        # Evict oldest if at capacity
        self._evict_source_templates()

        tpl_path = self._templates_dir / f"tpl-{commit[:12]}"
        if tpl_path.exists():
            shutil.rmtree(tpl_path, ignore_errors=True)

        # Unlock any stale lock and prune missing worktrees
        self._run(
            ["git", "worktree", "unlock", str(tpl_path)],
            cwd=self._repo, timeout=10, check=False,
        )
        self._run(
            ["git", "worktree", "prune"],
            cwd=self._repo, timeout=30, check=False,
        )

        # Create worktree
        self._log(log_file,
                  f"  Creating worktree at {tpl_path.name}...\n")
        self._run(
            ["git", "worktree", "add", "--force", "--detach",
             str(tpl_path), commit],
            cwd=self._repo, timeout=120, log_file=log_file,
        )

        # Fast local submodule init
        self._init_submodules_local(tpl_path, log_file)

        # Lock worktree so git doesn't prune it
        self._run(
            ["git", "worktree", "lock", str(tpl_path)],
            cwd=self._repo, timeout=10, check=False,
        )

        with self._tpl_cache_lock:
            self._tpl_cache[commit] = {
                "path": tpl_path, "last_used": time.time()
            }
        self._log(log_file,
                  f"  Source template ready: {tpl_path}\n")
        return tpl_path

    def _init_submodules_local(self, worktree: Path, log_file=None):
        """
        Initialize submodules by copying .git/modules from the golden
        repo (local, no network), then running submodule update.
        """
        main_modules = self._repo / ".git" / "modules"
        if not main_modules.exists():
            # Fallback: remote fetch
            self._log(log_file,
                      "  No cached modules, fetching from remote...\n")
            cpu = os.cpu_count() or 4
            self._run(
                ["git", "submodule", "update", "--init", "--recursive",
                 "--depth", "1", f"--jobs={cpu}"],
                cwd=worktree, timeout=600, log_file=log_file,
            )
            return

        # Resolve worktree's actual gitdir
        gitfile = worktree / ".git"
        if gitfile.is_file():
            content = gitfile.read_text().strip()
            if content.startswith("gitdir: "):
                actual_gitdir = Path(content[8:])
                if not actual_gitdir.is_absolute():
                    actual_gitdir = (worktree / actual_gitdir).resolve()
            else:
                actual_gitdir = gitfile
        else:
            actual_gitdir = gitfile

        modules_dest = actual_gitdir / "modules"
        self._log(log_file,
                  "  Copying submodule objects from golden repo (local)...\n")
        rc, _ = self._run(
            ["cp", "-a", "--reflink=auto",
             str(main_modules), str(modules_dest)],
            timeout=120, check=False,
        )
        if rc != 0:
            logger.warning("Module copy failed, falling back to remote")

        self._log(log_file,
                  "  Initializing submodule working trees...\n")
        self._run(
            ["git", "submodule", "update", "--init", "--recursive"],
            cwd=worktree, timeout=600, log_file=log_file,
        )

    def _evict_source_templates(self):
        """Evict LRU source templates if over capacity."""
        with self._tpl_cache_lock:
            while len(self._tpl_cache) >= self._max_source_tpl:
                oldest = min(
                    self._tpl_cache,
                    key=lambda k: self._tpl_cache[k]["last_used"],
                )
                entry = self._tpl_cache.pop(oldest)
                logger.info("Evicting source template %s", oldest[:12])
                self._run(
                    ["git", "worktree", "unlock", str(entry["path"])],
                    cwd=self._repo, timeout=10, check=False,
                )
                rc, _ = self._run(
                    ["git", "worktree", "remove", "--force",
                     str(entry["path"])],
                    cwd=self._repo, timeout=60, check=False,
                )
                if rc != 0 and entry["path"].exists():
                    shutil.rmtree(entry["path"], ignore_errors=True)
                self._run(
                    ["git", "worktree", "prune"],
                    cwd=self._repo, timeout=30, check=False,
                )

    # ------------------------------------------------------------------
    # Build template cache  (configured + compiled)
    # ------------------------------------------------------------------

    @staticmethod
    def build_cache_key(commit: str, board: str,
                        vehicle_waf_cmd: str,
                        extra_hwdef_content: str) -> str:
        """Deterministic cache key for a build configuration."""
        hwdef_hash = hashlib.sha256(
            extra_hwdef_content.encode()
        ).hexdigest()
        raw = f"{commit}:{board}:{vehicle_waf_cmd}:{hwdef_hash}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get_or_create_build_template(
        self, commit: str, board: str, vehicle_waf_cmd: str,
        extra_hwdef_content: str, log_file=None,
        build_timeout: int = 900,
    ) -> Path:
        """
        Get a cached build template or create one by running
        waf configure + build.  Thread-safe with per-key locking.
        Returns path to the build template directory.
        """
        key = self.build_cache_key(
            commit, board, vehicle_waf_cmd, extra_hwdef_content
        )

        # Fast check
        with self._build_cache_lock:
            if key in self._build_cache:
                entry = self._build_cache[key]
                if entry["path"].exists():
                    entry["last_used"] = time.time()
                    self._log(
                        log_file,
                        f"Build cache HIT ({key[:8]}): "
                        f"skipping configure+build\n",
                    )
                    return entry["path"]
                del self._build_cache[key]

        # Per-key lock
        with self._build_cache_lock:
            if key not in self._build_key_locks:
                self._build_key_locks[key] = threading.Lock()
            lock = self._build_key_locks[key]

        with lock:
            # Double-check
            with self._build_cache_lock:
                if key in self._build_cache:
                    entry = self._build_cache[key]
                    if entry["path"].exists():
                        entry["last_used"] = time.time()
                        return entry["path"]
                    del self._build_cache[key]

            return self._create_build_template(
                key, commit, board, vehicle_waf_cmd,
                extra_hwdef_content, log_file, build_timeout,
            )

    def _create_build_template(
        self, key: str, commit: str, board: str,
        vehicle_waf_cmd: str, extra_hwdef_content: str,
        log_file=None, build_timeout: int = 900,
    ) -> Path:
        """Build a new template: copy source, configure, compile."""
        self._log(
            log_file,
            f"Build cache MISS ({key[:8]}): "
            f"building {vehicle_waf_cmd} for {board}...\n",
        )

        # Evict oldest if at capacity
        self._evict_build_templates()

        # Get source template
        source_tpl = self.get_or_create_source_template(commit, log_file)

        bld_path = (
            self._build_templates_dir
            / f"bld-{key}-{board.lower()}"
        )
        if bld_path.exists():
            shutil.rmtree(bld_path, ignore_errors=True)

        # Copy source template to build dir
        self._log(log_file, "Copying source template to build dir...\n")
        self._run(
            ["cp", "-a", "--reflink=auto",
             str(source_tpl), str(bld_path)],
            timeout=300,
        )

        # Write extra_hwdef.dat
        hwdef_path = bld_path / "extra_hwdef.dat"
        hwdef_path.write_text(extra_hwdef_content)

        # Configure
        self._log(log_file,
                  f"=== Configure: waf configure --board {board} ===\n")
        self._run(
            ["python3", "./waf", "configure",
             "--board", board,
             "--extra-hwdef", str(hwdef_path)],
            cwd=bld_path, timeout=build_timeout, log_file=log_file,
        )

        # Build
        cpu = os.cpu_count() or 4
        self._log(log_file,
                  f"=== Build: waf {vehicle_waf_cmd} -j{cpu} ===\n")
        self._run(
            ["python3", "./waf", vehicle_waf_cmd, f"-j{cpu}"],
            cwd=bld_path, timeout=build_timeout, log_file=log_file,
        )

        # Cache it
        with self._build_cache_lock:
            self._build_cache[key] = {
                "path": bld_path, "last_used": time.time()
            }
        self._log(log_file, f"Build template cached ({key[:8]})\n")
        return bld_path

    def _evict_build_templates(self):
        """Evict LRU build templates if over capacity."""
        with self._build_cache_lock:
            while len(self._build_cache) >= self._max_build_tpl:
                oldest = min(
                    self._build_cache,
                    key=lambda k: self._build_cache[k]["last_used"],
                )
                entry = self._build_cache.pop(oldest)
                logger.info("Evicting build template %s", oldest[:8])
                if entry["path"].exists():
                    shutil.rmtree(entry["path"], ignore_errors=True)

    # ------------------------------------------------------------------
    # Build working copy
    # ------------------------------------------------------------------

    def copy_build_output(self, build_template: Path,
                          board: str, dest_dir: Path):
        """
        Copy compiled binaries and build log from a build template
        to the per-build artifacts directory.
        """
        # The waf build output goes to build/<board>/bin/
        bin_src = build_template / "build" / board / "bin"
        bin_dest = dest_dir / board / "bin"
        if bin_src.exists():
            bin_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(bin_src, bin_dest, dirs_exist_ok=True)
        else:
            logger.warning("No bin dir at %s", bin_src)

    # ------------------------------------------------------------------
    # Subprocess helpers
    # ------------------------------------------------------------------

    def _run(self, cmd: list[str], cwd: str | Path | None = None,
             timeout: int = 120, check: bool = True,
             log_file=None) -> tuple[int, str]:
        """
        Run a subprocess, optionally streaming output to a log file.
        Returns (returncode, combined_output).
        """
        logger.debug("Running: %s", " ".join(cmd))
        proc = subprocess.Popen(
            cmd, cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        chunks = []
        try:
            deadline = time.time() + timeout
            for line in iter(proc.stdout.readline, b""):
                if time.time() > deadline:
                    proc.kill()
                    proc.wait()
                    msg = "Command timed out"
                    if log_file:
                        log_file.write(msg + "\n")
                        log_file.flush()
                    return -1, msg
                text = line.decode(errors="replace")
                chunks.append(text)
                if log_file:
                    log_file.write(text)
                    log_file.flush()
            proc.wait()
        except Exception:
            proc.kill()
            proc.wait()
            raise

        output = "".join(chunks)
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode, cmd, output=output
            )
        return proc.returncode, output

    @staticmethod
    def _log(log_file, msg: str):
        """Write a message to log_file if provided."""
        if log_file:
            log_file.write(msg)
            log_file.flush()
