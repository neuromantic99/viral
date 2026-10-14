from pathlib import Path

HERE = Path(__file__).parent

from viral.models import GrosmarkConfig

SERVER_PATH = Path("YOUR/SERVER/FOLDER")


# The below are probably true of all server paths but may need adjusting
BEHAVIOUR_DATA_PATH = SERVER_PATH / Path("Behaviour/online/Subjects")
SYNC_FILE_PATH = SERVER_PATH / "DAQami"
TIFF_UMBRELLA = SERVER_PATH / "2P"

CACHE_PATH = SERVER_PATH / "viral_caches" / "cached_2p"
TEMP_CACHE_PATH = SERVER_PATH / "viral_caches" / "temp_caches"

# Probably true of all encoders but again may need adjusting
ENCODER_TICKS_PER_TURN = 360

LOCAL_DFF_PATH = Path("YOUR/LOCAL/FOLDER")


# The ID of your google sheet can be found in the URL:
# https://docs.google.com/spreadsheets/d/ID-GOES-HERE

SPREADSHEET_ID = "YOUR-SPREADSHEET-ID"

# Add your config for the grosmark analysis here
grosmark_config = GrosmarkConfig(
    bin_size=2,
    start=0,
    end=180,
)
