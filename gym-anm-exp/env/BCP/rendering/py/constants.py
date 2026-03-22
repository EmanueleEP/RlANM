import os
 
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RENDERING_FOLDER = os.path.dirname(_THIS_DIR)
RENDERING_LOGS = os.path.join(RENDERING_FOLDER, "logs")
RENDERING_RELATIVE_PATH = os.path.join("env", "BCP", "rendering")
 
ROOT_FOLDER = os.path.normpath(os.path.join(_THIS_DIR, "..", "..", "..", ".."))
 
if not os.path.isdir(ROOT_FOLDER):
    raise RuntimeError(
        f"ROOT_FOLDER not found: {ROOT_FOLDER}\n"
        "Check the file structure, it should look like this:\n"
        "gym-anm-exp/env/BCP/rendering/py/constants.py"
    )
 