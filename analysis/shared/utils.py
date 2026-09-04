import pycountry

def iso_num_to_alpha3(num):
    try:
        c = pycountry.countries.get(numeric=str(int(num)).zfill(3))
        return c.alpha_3 if c else str(int(num))
    except Exception:
        return str(int(num))
