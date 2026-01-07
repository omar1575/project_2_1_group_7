import config


REQUIRED_SETTINGS = ["DATA_FILENAME"]

def test():

    missing = []
    for key in REQUIRED_SETTINGS:
        if key not in config.SETTINGS or not config.SETTINGS[key]:
            missing.append(key)

    if missing:
        raise RuntimeError(f"Missing config keys: {missing}")