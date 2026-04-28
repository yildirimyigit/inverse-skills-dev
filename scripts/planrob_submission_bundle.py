from __future__ import annotations

import json

from inverse_skills.planrob_bundle import build_bundle, write_bundle_artifacts


def main() -> None:
    bundle = build_bundle()
    print(json.dumps(bundle, indent=2))
    json_path, md_path, tex_path = write_bundle_artifacts(bundle)
    print(f"\nSaved {json_path}")
    print(f"Saved {md_path}")
    print(f"Saved {tex_path}")


if __name__ == "__main__":
    main()
