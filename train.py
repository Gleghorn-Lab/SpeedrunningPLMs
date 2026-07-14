import entrypoint_setup  # noqa: F401

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from speedrunning_plms.training.cli import main


if __name__ == "__main__":
    main()
