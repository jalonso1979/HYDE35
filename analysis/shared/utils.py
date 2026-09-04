import pycountry

def fix_iso3(val):
    if str(val).isnumeric():
        try:
            c = pycountry.countries.get(numeric=str(val).zfill(3))
            return c.alpha_3 if c else val
        except Exception:
            return val
    return val
