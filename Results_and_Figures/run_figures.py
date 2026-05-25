from pathlib import Path
import os
import subprocess
import sys


# Edit this tuple to run a subset, for example ("Figs_L96",).
FOLDERS_TO_RUN = ("Figs_L96", "Figs_Kol", "Figs_CLV")
STOP_ON_ERROR = True

SCRIPTS_BY_FOLDER = {
    "Figs_L96": (
        "plot_fig3.py",
        "plot_fig4_5.py",
        "plot_fig6.py",
        "plot_fig7_8.py",
        "plot_fig9.py",
        "plot_fig10.py",
        "plot_fig11.py",
    ),
    "Figs_Kol": (
        "plot_fig12.py",
        "plot_fig13.py",
        "plot_fig14.py",
        "plot_fig15.py",
        "plot_fig16.py",
        "plot_fig17.py",
        "plot_fig18.py",
    ),
    "Figs_CLV": (
        "plot_fig20.py",
        "plot_fig21.py",
        "plot_fig22.py",
        "plot_fig23.py",
        "plot_fig24.py",
        "plot_fig25.py",
    ),
}


def run_script(folder, script_name):
    folder_path = Path(folder)
    script_path = folder_path / script_name
    if not script_path.exists():
        raise FileNotFoundError(script_path)

    env = os.environ.copy()
    mpl_cache = Path("/tmp/matplotlib-figures-cache")
    mpl_cache.mkdir(exist_ok=True)
    env.setdefault("MPLCONFIGDIR", str(mpl_cache))

    print(f"Running {script_path}", flush=True)
    return subprocess.run([sys.executable, script_name], cwd=folder_path, env=env)


def main():
    failures = []
    for folder in FOLDERS_TO_RUN:
        for script_name in SCRIPTS_BY_FOLDER[folder]:
            result = run_script(folder, script_name)
            if result.returncode != 0:
                failures.append((folder, script_name, result.returncode))
                if STOP_ON_ERROR:
                    break
        if failures and STOP_ON_ERROR:
            break

    if failures:
        print("\nFailed scripts:")
        for folder, script_name, returncode in failures:
            print(f"  {folder}/{script_name}: return code {returncode}")
        return 1

    print("\nAll selected figure scripts finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
