# run.py <script> : run a figure script four times, once per web variant (webstyle patches savefig)
import sys, os, subprocess
if os.environ.get("WEBFIG_VARIANT"):
    import runpy
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import webstyle  # noqa: F401
    script = os.path.abspath(sys.argv[1])
    os.chdir(os.path.dirname(script)); sys.path.insert(0, os.path.dirname(script)); sys.argv = [script]
    runpy.run_path(script, run_name="__main__")
else:
    for v in ("wide-light", "wide-dark", "narrow-light", "narrow-dark"):
        subprocess.run([sys.executable, os.path.abspath(__file__), sys.argv[1]], check=True,
                       env={**os.environ, "WEBFIG_VARIANT": v})
