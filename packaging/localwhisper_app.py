"""PyInstaller entry script for LocalWhisper.app: the menu bar app, nothing else.

Kept outside the package so PyInstaller analyses `localwhisper` as a normal
import (relative imports inside `gui/` need a parent package).
"""

import sys

from localwhisper.gui.menubar import main

sys.exit(main())
