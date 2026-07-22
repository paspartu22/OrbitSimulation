import warnings
from ssl_bootstrap import configure_ssl_certificates

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API.*",
    category=UserWarning,
)

configure_ssl_certificates()
from pyatmos import download_sw_jb2008,read_sw_jb2008
# Download or update the space weather file from https://sol.spacenvironment.net
swfile = download_sw_jb2008() 
# Read the space weather data
swdata = read_sw_jb2008(swfile)

from pyatmos import jb2008
# Set a specific time and location
t = '2026-06-05 22:18:45' # time(UTC)
lat,lon,alt = 25,102,280 # latitude, longitude in [degree], and altitude in [km]
jb08 = jb2008(t,(lat,lon,alt),swdata)
print(jb08.rho) # [kg/m^3]
print(jb08.T) # [K]