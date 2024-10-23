from pathlib import Path
import math


HERE = Path(__file__).parent

SERVER_PATH = Path("/Volumes/MarcBusche/James/")

BEHAVIOUR_DATA_PATH = SERVER_PATH / Path("Behaviour/online/Subjects")

WHEEL_CIRCUMFERENCE = 11 * math.pi  # cm
print(f"WHEEL_CIRCUMFERENCE: {WHEEL_CIRCUMFERENCE}")
ENCODER_TICKS_PER_TURN = 360  # check me

SPREADSHEET_ID = "1fMnVXrDeaWTkX-TT21F4mFuAlaXIe6uVteEjv8mH0Q4"
