#!/usr/bin/env python3
"""
upload_to_HF.py — Upload a folder to Hugging Face Hub with auto repo creation & smart auth.

Examples:
  # Basic (uses cached login or env token)
  python upload_to_HF.py --repo_id todd427/toddric-3b-merged-v3 --folder ./out --repo_type model

  # Private repo, custom commit message, upload to a branch
  python upload_to_HF.py --repo_id org/proj --folder ./dist --repo_type model --private \
                  --branch main --message "Initial upload"

  # Dataset with include/exclude globs
  python upload_to_HF.py --repo_id todd427/my-ds --repo_type dataset --folder ./data \
                  --allow "*.parquet" --ignore "*.tmp"

  # Space (must specify SDK)
  python upload_to_HF.py --repo_id todd427/my-space --repo_type space --folder ./app --space_sdk gradio
"""

import os
import argparse
from typing import Optional, List

from huggingface_hub import HfApi, upload_folder, whoami
from huggingface_hub.utils import HfHubHTTPError


def resolve_token(cli_token: Optional[str]) -> Optional[str]:
    """
    Priority: CLI --token > env HUGGINGFACE_HUB_TOKEN > cached login (None here, library will handle)
    """
    if cli_token:
        return cli_token
    env_token = os.getenv("HUGGINGFACE_HUB_TOKEN")
    if env_token:
        return env_token
    # No explicit token; huggingface_hub will use cached login if present.
    return None


def assert_auth_or_hint(api: HfApi, token: Optional[str]):
    """
    Try a lightweight call to verify we can talk to the Hub with current auth.
    If not authenticated and repo is private, operations will fail—so give a helpful hint.
    """
    try:
        _ = whoami(token=token)  # succeeds if logged in or token valid; raises otherwise
    except Exception:
        # Do not hard-fail; public repos still work. Provide a helpful note instead.
        print("ℹ️  No valid auth token found. If you hit auth errors (private/gated repos), run:")
        print("    huggingface-cli login")
        print("    # or set env var: export HUGGINGFACE_HUB_TOKEN=hf_...")


def ensure_repo_exists(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    private: bool,
    space_sdk: Optional[str],
    token: Optional[str],
):
    """
    Check if repo exists; if not, create it.
    - For Spaces, you can pass --space_sdk (gradio/streamlit/docker)
    """
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type, token=token)
        print(f"✔ Repo exists: {repo_type}:{repo_id}")
    except HfHubHTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            print(f"✚ Creating repo {repo_type}:{repo_id} (private={private}) ...")
            create_kwargs = dict(
                repo_id=repo_id,
                repo_type=repo_type,
                private=private,
                exist_ok=True,
                token=token,
            )
            if repo_type == "space" and space_sdk:
                create_kwargs["space_sdk"] = space_sdk
            api.create_repo(**create_kwargs)
            print("✔ Created.")
        else:
            raise


def main():
    p = argparse.ArgumentParser(description="Upload a folder to Hugging Face Hub.")
    p.add_argument("--repo_id", required=True, help="e.g., username/my-model or org/my-model")
    p.add_argument("--repo_type", choices=["model", "dataset", "space"], default="model")
    p.add_argument("--folder", required=True, help="Local folder to upload")
    p.add_argument("--private", action="store_true", help="Create as private (if creating)")
    p.add_argument("--branch", default="main", help="Target branch (revision) to commit to")
    p.add_argument("--message", default="Upload via upload_to_HF.py", help="Commit message")
    p.add_argument("--allow", nargs="*", default=None, help="Allow patterns (globs), e.g. *.safetensors *.json")
    p.add_argument("--ignore", nargs="*", default=None, help="Ignore patterns (globs), e.g. *.tmp *.log")
    p.add_argument("--token", default=None, help="Hugging Face token (else uses env/cached login)")
    p.add_argument("--space_sdk", choices=["gradio", "streamlit", "docker"], default=None,
                   help="For repo_type=space only")
    args = p.parse_args()

    token = resolve_token(args.token)
    api = HfApi(token=token)

    # Soft-check auth and give hints
    assert_auth_or_hint(api, token)

    # Ensure repo exists (create if missing)
    ensure_repo_exists(
        api=api,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        private=args.private,
        space_sdk=args.space_sdk,
        token=token,
    )

    # Upload the folder
    print(f"⬆️  Uploading '{args.folder}' → {args.repo_type}:{args.repo_id}@{args.branch}")
    upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=args.folder,
        revision=args.branch,
        commit_message=args.message,
        allow_patterns=args.allow,
        ignore_patterns=args.ignore,
        token=token,           # explicit so CI works even without cached login
        run_as_future=False,   # perform upload now
    )
    print("✅ Upload complete.")


if __name__ == "__main__":
    main()
