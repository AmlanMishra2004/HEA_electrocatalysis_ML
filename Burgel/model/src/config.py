import os

R_GAS_CONSTANT = 8.314462618  # J/(mol*K)
OER_REFERENCE_POTENTIAL_V = 1.23
TARGET_CURRENT_A_CM2 = 0.01  # 10 mA/cm^2

SCHERRER_K = 0.9
XRD_WAVELENGTH_ANGSTROM = 1.5406  # Cu K-alpha
DEFAULT_MILLER_FCC = (1, 1, 1)

SAVGOL_WINDOW = 21
SAVGOL_POLYORDER = 2

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.10
TEST_RATIO = 0.20

MASTER_ELEMENTS = ["Ni", "Pd", "Pt", "Ru", "Ir", "Cu"]
NOBLE_ELEMENTS = ["Pd", "Pt", "Ru", "Ir"]
NON_NOBLE_ELEMENTS = ["Ni", "Cu"]

MIN_VALID_ETA = -1.0
MAX_VALID_ETA = 2.0

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "..", "15453660")
DEFAULT_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
DEFAULT_PROPERTIES_PATH = os.path.join(PROJECT_ROOT, "element_properties.json")
