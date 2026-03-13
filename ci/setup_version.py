"""Add pre-release version to the package.

This script is used to set the pre-release version for beta releases
(e.g. 0.10.17-beta.1) when the tag indicates a beta release.

With the new automated release process, stable versions are already 
updated by bump-my-version during the release workflow.
"""

from packaging.version import parse
import argparse


def main():
    parser = argparse.ArgumentParser(description="Set the version of the package.")
    parser.add_argument("version", type=str, help="The version to set.")
    args = parser.parse_args()

    # Determine the target version string (strip 'v' if present)
    target_version = args.version
    if target_version.startswith('v'):
        target_version = target_version[1:]

    with open("python/Cargo.toml", "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("version = "):
                current_version = line.split('"')[1]
                lines[i] = f'version = "{target_version}"\n'
                break
        else:
            raise ValueError("Could not find version in Cargo.toml")

    from packaging.version import InvalidVersion
    try:
        parsed_version = parse(target_version)
        current_version_parsed = parse(current_version)

        # For beta releases, we expect the base version to be the next version
        # For example, if current is 0.10.17 and we're releasing 0.10.18-beta.1
        # This is different from the old process where the version was not bumped yet
        if parsed_version.is_prerelease:
            # Just ensure it's a valid version transition
            print(f"Setting beta version: {current_version} -> {target_version}")
        else:
            # For stable releases, version should already match
            assert (
                parsed_version.release == current_version_parsed.release
            ), f"Version mismatch for stable release: {parsed_version.release} != {current_version_parsed.release}"
    except InvalidVersion:
        print(f"Setting custom version (non-PEP 440): {current_version} -> {target_version}")

    with open("python/Cargo.toml", "w") as f:
        f.writelines(lines)


if __name__ == "__main__":
    main()
