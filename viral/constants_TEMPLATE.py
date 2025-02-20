from pathlib import Path


HERE = Path(__file__).parent

SERVER_PATH = Path("YOUR/SERVER/FOLDER")


# The below are probably true of all server paths but may need adjusting
BEHAVIOUR_DATA_PATH = SERVER_PATH / Path("Behaviour/online/Subjects")
SYNC_FILE_PATH = SERVER_PATH / "DAQami"
TIFF_UMBRELLA = SERVER_PATH / "2P"

# Probably true of all encoders but again may need adjusting
ENCODER_TICKS_PER_TURN = 360


# The ID of your google sheet can be found in the URL:
# https://docs.google.com/spreadsheets/d/ID-GOES-HERE

SPREADSHEET_ID = "YOUR-SPREADSHEET-IDk"
