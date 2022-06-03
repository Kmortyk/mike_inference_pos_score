from .config import *

# configures project
import sys
sys.path.append(TFOD2_PATH)
sys.path.append(TFOD2_SLIM_PATH)

# disable tf logs
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
