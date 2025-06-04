import datetime
import pytz

def get_utc_time():
    """Returns the current time in UTC format."""
    return datetime.datetime.now(datetime.timezone.utc)

def convert_to_user_tz(dt: datetime.datetime, user_timezone: str) -> datetime.datetime:
    """
    Convert a UTC datetime object to a given user's timezone and return as datetime.

    Args:
        dt (datetime.datetime): The datetime object in UTC.
        user_timezone (str): The user's timezone.

    Returns:
        datetime.datetime: The datetime object in the user's timezone.
    """

    timezone = pytz.timezone(user_timezone) if user_timezone in pytz.all_timezones else pytz.utc
    return dt.replace(tzinfo=pytz.utc).astimezone(timezone)
