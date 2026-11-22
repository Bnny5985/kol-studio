"""Entry point for the bundled app (PyInstaller)."""
import multiprocessing
import os
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # A bundled app has no terminal: keep library logs from blocking on a closed stdout.
    if getattr(sys, "frozen", False) and sys.stdout is None:
        sys.stdout = sys.stderr = open(os.devnull, "w")
    from kol_studio.desktop import main

    main()
