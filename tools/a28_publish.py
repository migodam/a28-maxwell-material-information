"""Authorized A28 E1 publication, independent of scientific and budget status.

This source does not itself run physics. Invocation requires reviewed README.md,
E1_DECISION.md and saved artifacts. No locked/TRAIN payload, keys or hash checks.
"""
from __future__ import annotations

import time
START_WALL, START_CPU = time.perf_counter(), time.process_time()
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
from a28.budget import charged_job, write_json
from a28_prepare_delivery import (check_frozen_delivery, public_file, raw_bundle,
                                  register_external_audits, render_accounting)

REPO = "migodam/a28-maxwell-material-information"
TAG = "a28-e1-v0.1.0"
REMOTE = "a28"
REPO_URL = "https://github.com/" + REPO
PUSH_URL = REPO_URL + ".git"


def _timeout(book, proposed):
    book.check()
    remaining = book.deadline - time.perf_counter()
    if remaining <= 2:
        raise RuntimeError("PUBLICATION_TIME_RESERVE_EXHAUSTED")
    return min(float(proposed), remaining - 1)


def command(args, book, timeout=120, *, allow_failure=False):
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True,
                            timeout=_timeout(book, timeout))
    book.check()
    if result.returncode and not allow_failure:
        raise RuntimeError("PUBLICATION_COMMAND_FAILED:" + args[0] + "; private diagnostic output suppressed")
    return result


def anonymous(url, book, *, needle=None, zip_asset=False):
    """Use no auth header/cookie; check actual immutable report and asset URLs."""
    request = urllib.request.Request(url, headers={"User-Agent": "A28-anonymous-public-artifact-check"})
    with urllib.request.urlopen(request, timeout=_timeout(book, 25)) as response:
        value = response.read(500000 if needle is not None else 64)
        status = response.status
        final_url = response.geturl()
    if status != 200 or (needle is not None and needle.encode("utf-8") not in value):
        raise ValueError("ANONYMOUS_CONTENT_VERIFICATION_FAILED")
    if zip_asset and not value.startswith(b"PK\x03\x04"):
        raise ValueError("ANONYMOUS_RELEASE_ASSET_IS_NOT_ZIP")
    book.check()
    return {"url": url, "status": status, "authentication_sent": False,
            "expected_content_checked": needle is not None, "zip_header_checked": zip_asset,
            "redirected": final_url != url}


def _permitted_stage_name(name):
    path = Path(name)
    if name in {".gitignore", "README.md", "E1_DECISION.md", "a28_config.json",
                "research/A28_E1_PRE_RUN_PLAN.md"}:
        return True
    if len(path.parts) == 1 and path.name.startswith("A28_") and path.suffix in {".md", ".json", ".jsonl", ".csv"}:
        return True
    if name.startswith(("src/a28/", "results/a28/", "theory/a28/", "research/delegated/a28_")):
        return True
    if len(path.parts) == 2 and path.parts[0] == "tools" and path.name.startswith("a28") and path.suffix == ".py":
        return True
    if len(path.parts) == 2 and path.parts[0] == "tests" and path.name.startswith("test_a28") and path.suffix == ".py":
        return True
    return name == "publication/A28_PUBLIC_ARTIFACT_AUDIT.json"


def stage_public_source(book):
    paths = [".gitignore", "README.md", "E1_DECISION.md", "a28_config.json", "src/a28", "results/a28", "theory/a28",
             "research/A28_E1_PRE_RUN_PLAN.md"]
    paths += [path.name for path in ROOT.glob("A28_*") if path.is_file() and path.suffix in {".md", ".json", ".jsonl", ".csv"}]
    paths += [path.relative_to(ROOT).as_posix() for path in (ROOT / "tools").glob("a28*.py")]
    paths += [path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").glob("test_a28*.py")]
    paths += [path.relative_to(ROOT).as_posix() for path in (ROOT / "research/delegated").glob("a28_*") if path.is_dir()]
    command(["git", "add", "--", *paths], book)
    names = [name for name in command(["git", "diff", "--cached", "--name-only", "-z"], book).stdout.split("\0") if name]
    rejected = []
    for name in names:
        if not _permitted_stage_name(name):
            rejected.append(name)
            continue
        path = ROOT / name
        # A deletion must still belong to this project's permitted path list.
        if not path.exists():
            continue
        public_file(path)
    if rejected:
        raise ValueError("UNRELATED_STAGED_FILES:" + json.dumps(sorted(set(rejected))))
    write_json(ROOT / "publication/A28_PUBLIC_ARTIFACT_AUDIT.json", {
        "status": "PASS", "files": len(names), "secret_scan": "matched bytes never logged",
        "training_payload": "EXCLUDED", "locked_payload": "EXCLUDED", "exact_state_LU_caches": "EXCLUDED",
        "binary_scientific_arrays": "Release only", "SHA256_checks": "NOT_RUN",
        "this_status_is_publication_safety_only": True, "scientific_status": "E1_DECISION.md"})
    command(["git", "add", "publication/A28_PUBLIC_ARTIFACT_AUDIT.json"], book)
    source_names = [name for name in names if name.endswith(".py")]
    if source_names:
        command(["git", "diff", "--cached", "--check", "--", *source_names], book)
    diff = command(["git", "diff", "--cached", "--quiet"], book, allow_failure=True)
    if diff.returncode == 1:
        command(["git", "commit", "-m", "A28 E1: material-information diagnosis with retained failures and strict budget audit"], book)
    elif diff.returncode != 0:
        raise RuntimeError("STAGED_DIFF_INSPECTION_FAILED")
    return command(["git", "rev-parse", "HEAD"], book).stdout.strip()


def ensure_public_destination(book, sha):
    probe = command(["gh", "repo", "view", REPO, "--json", "url,isEmpty,isPrivate"], book, allow_failure=True)
    remotes = command(["git", "remote"], book).stdout.splitlines()
    existing_remote = REMOTE in remotes
    if existing_remote and command(["git", "remote", "get-url", REMOTE], book).stdout.strip() != PUSH_URL:
        raise ValueError("UNEXPECTED_A28_PUSH_DESTINATION")
    if probe.returncode:
        command(["gh", "repo", "create", REPO, "--public", "--description",
                 "A28 E1 exploratory Maxwell material-information diagnosis; strict full-wave cap violation disclosed"], book)
    else:
        existing = json.loads(probe.stdout)
        if existing.get("isPrivate") or existing.get("url") != REPO_URL:
            raise ValueError("PUBLIC_A28_DESTINATION_MISMATCH")
        if not existing.get("isEmpty") and not existing_remote:
            raise ValueError("EXISTING_NONEMPTY_REPOSITORY_WITHOUT_OWNED_REMOTE")
    if not existing_remote:
        command(["git", "remote", "add", REMOTE, PUSH_URL], book)
    refs = command(["git", "ls-remote", REMOTE, "refs/heads/main"], book).stdout.split()
    if refs and refs[0] != sha:
        # Retry/update only within the existing local ancestry; never force push.
        ancestry = command(["git", "merge-base", "--is-ancestor", refs[0], sha], book, allow_failure=True)
        if ancestry.returncode:
            raise ValueError("REMOTE_MAIN_IS_NOT_LOCAL_ANCESTOR; no overwrite")


def _exposed_asset_requested(include):
    path = ROOT / "publication/A28_E1_EXPOSED_INPUTS.zip"
    if not include:
        return None
    if not path.is_file() or path.is_symlink():
        raise ValueError("EXPLICIT_EXPOSED_INPUT_BUNDLE_NOT_PREPARED")
    with zipfile.ZipFile(path) as archive:
        scope = json.loads(archive.read("INPUT_SCOPE.json"))
        if (scope.get("status") != "EXPOSED_DEVELOPMENT_ONLY" or
                scope.get("TRAIN_payload") != "EXCLUDED" or scope.get("locked_ID_OOD_payload") != "EXCLUDED"):
            raise ValueError("EXPOSED_INPUT_SCOPE_MISMATCH")
        for name in archive.namelist():
            parts = Path(name).parts
            if (name.startswith(("/", "\\")) or ".." in parts or
                    any("locked" in part.lower() or part.lower() == "private" or
                        part.lower().startswith("train.") or part == "exact_states" for part in parts)):
                raise ValueError("FORBIDDEN_INPUT_RELEASE_MEMBER")
    return path


def release_assets(book, sha, files, notes):
    probe = command(["gh", "release", "view", TAG, "--repo", REPO, "--json", "url,assets,targetCommitish"],
                    book, allow_failure=True)
    if probe.returncode:
        command(["gh", "release", "create", TAG, *(str(path) for path in files), "--repo", REPO,
                 "--target", sha, "--title", "A28 E1 material-information v0.1.0", "--notes-file", str(notes)],
                book, timeout=240)
    else:
        # Existing immutable tag must already target exactly this commit.
        refs = command(["git", "ls-remote", REMOTE, "refs/tags/" + TAG, "refs/tags/" + TAG + "^{}"], book).stdout
        mapping = dict((line.split()[1], line.split()[0]) for line in refs.splitlines() if line.strip())
        tag_sha = mapping.get("refs/tags/" + TAG + "^{}", mapping.get("refs/tags/" + TAG))
        if tag_sha != sha:
            raise ValueError("EXISTING_RELEASE_TAG_TARGET_MISMATCH; no retag")
        existing_names = {asset["name"] for asset in json.loads(probe.stdout).get("assets", [])}
        missing = [str(path) for path in files if path.name not in existing_names]
        if missing:
            command(["gh", "release", "upload", TAG, *missing, "--repo", REPO], book, timeout=240)
    return json.loads(command(["gh", "release", "view", TAG, "--repo", REPO,
                              "--json", "url,assets,targetCommitish"], book).stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("--include-exposed-inputs", action="store_true")
    args = parser.parse_args()
    register_external_audits()
    config = json.loads((ROOT / "a28_config.json").read_text(encoding="utf-8"))
    with charged_job(ROOT, args.job, config=config, metadata={"stage": "public-delivery"}) as book:
        book.deadline -= book.started_wall - START_WALL
        book.started_wall, book.started_cpu = START_WALL, START_CPU
        check_frozen_delivery(book)
        for path in [*(ROOT / "src/a28").glob("*.py"), *(ROOT / "tools").glob("a28*.py"),
                     *(ROOT / "tests").glob("test_a28*.py")]:
            ast.parse(path.read_text(encoding="utf-8"), filename=path.name)
        costs = render_accounting("science-and-preparation-complete-before-publication", active_job=args.job)
        raw, members = raw_bundle(book, active_job=args.job)
        source = json.loads((ROOT / "A28_SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
        source.update(raw_bytes=raw.stat().st_size, archive_members=members,
                      active_publication_job_excluded_from_prepublication_raw=True)
        write_json(ROOT / "A28_SOURCE_MANIFEST.json", source)
        inputs = _exposed_asset_requested(args.include_exposed_inputs)
        files = [raw] + ([inputs] if inputs else [])
        sha = stage_public_source(book)
        ensure_public_destination(book, sha)
        print(json.dumps({"stage": "PUSH", "commit": sha,
                          "full_wave_budget_compliance": costs["full_wave_budget_compliance"]}), flush=True)
        command(["git", "push", REMOTE, "HEAD:main"], book, timeout=240)
        notes = ROOT / "publication/A28_RELEASE_NOTES.md"
        notes.write_text("# A28 E1 exploratory material-information diagnosis\n\n"
            "Read README.md and E1_DECISION.md for material counterfactuals, matched six-attempt headroom, "
            "finite conditional constitutive identities, BP information audit and scientific conclusions. "
            "No new NN training, locked evaluation, E2 or E3 was executed.\n\n"
            f"**Budget compliance FAIL:** campaign central states={costs['campaign_full_wave_material_states']}; "
            f"legacy real tiny fixtures={costs['legacy_unreserved_real_fixture_states']}; strict total "
            f"{costs['strict_state_count_label']} {costs['strict_all_resolution_full_wave_material_states']} "
            f"against cap {costs['full_wave_material_states_cap']}. Independent dense-reference audits make "
            "this a known lower bound. Historical receipts and frozen config are unchanged. "
            "No scientific GO authorizes protocol acceptance or further experiments.\n\n"
            "A28_E1_RAW_EVIDENCE.zip includes scientific NPZ/tables/figures and original job/action/failure records, "
            "excluding runtime exact-state LU caches. Final publication verification and completed "
            "publication accounting are a separate sidecar with an explicit cutoff. "
            "An optional exposed-input bundle contains only existing development inputs and frozen N3 dependencies.\n",
            encoding="utf-8")
        release = release_assets(book, sha, files, notes)
        remote_rows = command(["git", "ls-remote", REMOTE, "refs/heads/main", "refs/tags/" + TAG,
                               "refs/tags/" + TAG + "^{}"], book).stdout
        refs = dict((line.split()[1], line.split()[0]) for line in remote_rows.splitlines() if line.strip())
        tag_sha = refs.get("refs/tags/" + TAG + "^{}", refs.get("refs/tags/" + TAG))
        if refs.get("refs/heads/main") != sha or tag_sha != sha:
            raise ValueError("REMOTE_MAIN_OR_RELEASE_COMMIT_MISMATCH")
        urls = [REPO_URL, REPO_URL + "/blob/" + sha + "/E1_DECISION.md",
                "https://raw.githubusercontent.com/" + REPO + "/" + sha + "/README.md",
                "https://raw.githubusercontent.com/" + REPO + "/" + sha + "/E1_DECISION.md",
                "https://raw.githubusercontent.com/" + REPO + "/" + sha + "/a28_config.json", release["url"]]
        checks = [anonymous(url, book, needle="E1" if "raw." in url and url.endswith(".md") else
                            config["protocol_id"] if url.endswith("a28_config.json") else None) for url in urls]
        release_names = {asset["name"]: asset for asset in release.get("assets", [])}
        for path in files:
            if path.name not in release_names or int(release_names[path.name].get("size", -1)) != path.stat().st_size:
                raise ValueError("ACTUAL_RELEASE_ASSET_MISSING_OR_SIZE_MISMATCH")
            asset_url = REPO_URL + "/releases/download/" + TAG + "/" + urllib.parse.quote(path.name)
            checks.append(anonymous(asset_url, book, zip_asset=True))
        verification = {"publication": "VERIFIED", "repository": REPO_URL, "commit": sha,
            "remote_main_commit": refs["refs/heads/main"], "release_commit": tag_sha, "release": release["url"],
            "report": REPO_URL + "/blob/" + sha + "/E1_DECISION.md",
            "anonymous_raw_report": "https://raw.githubusercontent.com/" + REPO + "/" + sha + "/E1_DECISION.md",
            "anonymous_checks": checks, "scientific_decision": "Read E1_DECISION.md; independent from publication",
            "protocol_acceptance": costs["protocol_acceptance"], "full_wave_budget_compliance": costs["full_wave_budget_compliance"],
            "strict_full_wave_state_count_label": costs["strict_state_count_label"],
            "strict_full_wave_material_states": costs["strict_all_resolution_full_wave_material_states"],
            "new_training": "NOT_RUN", "new_labels": 0, "locked_ID_OOD": "UNOPENED", "E2_E3": "NOT_RUN",
            "SHA256_checks": "NOT_RUN", "accounting_cutoff": "completed receipt written on context exit"}
        write_json(ROOT / "publication/A28_PUBLICATION_VERIFICATION.json", verification)
        print(json.dumps(verification, ensure_ascii=False), flush=True)
    # Completed receipt is immutable. This accounting tail and any subsequent
    # sidecar upload are paid CPU jobs with explicit active-job cutoffs.
    tail_job = args.job + "-accounting"
    with charged_job(ROOT, tail_job, config=config, metadata={"stage": "accounting"}):
        render_accounting("publication-and-anonymous-primary-asset-verification-complete", active_job=tail_job)


if __name__ == "__main__":
    main()
