import logging
import sys
from datetime import datetime

from pytz import timezone, utc


def setup_logging_with_pacific_tz(prefix=""):
    logging.basicConfig(
        format=f"%(asctime)s %(levelname).1s {prefix}%(message)s",
        datefmt="%m/%d %I:%M:%S %p",
        stream=sys.stdout,
        level="INFO",
    )

    def pacific_time(*args):
        utc_dt = utc.localize(datetime.utcnow())
        my_tz = timezone("US/Pacific")
        converted = utc_dt.astimezone(my_tz)
        return converted.timetuple()

    logging.Formatter.converter = pacific_time
