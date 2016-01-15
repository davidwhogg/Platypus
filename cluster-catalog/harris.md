Cluster catalog
---------------

Things you want to know.

The Harris catalog is in a bad format (ASCII; disjoint). I've committed a script that will parse the ASCII format and collect all information from a cluster into a single row. That table has been written to `harris_mwgc.fits`

***Load data***

```python
  from astropy.table import Table
  t = Table.read("cluster-catalog/harris_mwgc.fits")
  
  # Print the columns available
  print(t.dtype.names)
```

***Column description***
```
ID          : Cluster identification number
Name        : Other commonly used cluster name
RA          : Right acsension (epoch J2000; HH:MM:SS)
DEC         : Declination (epoch J2000; +HH:MM:SS)
l           : Galactic longitude [deg]
b           : Galactic latitude [deg]
R_Sun       : Distance from Sun [kpc]
R_gc        : Distance from galactic center assuming R_0 = 8.0 kpc [kpc]
X           : Galactic X distance [kpc]
Y           : Galactic Y distance [kpc]
Z           : Galactic Z distance [kpc]

[Fe/H]      : Average cluster metallicity [dex]
wt          : "Weight" of mean metallicity; ~the number of "independent" [Fe/H] measurements
E(B-V)      : Foreground reddening 
V_HB        : Horizontal branch magnitude in V band
(m-M)V      : Apparant visual distance modulus
V_t         : Integrated V magnitude of the cluster
M_V,t       : Absolute visual magnitude (cluster luminosity), M_V,t = V_t - (m-M)V
U-B         : U-B color, uncorrected for reddening
B-V         : B-V color, uncorrected for reddening
V-R         : V-R color, uncorrected for reddening
V-I         : V-I color, uncorrected for reddening
spt         : Spectral type of the integrated cluster light
ellip       : Projected ellipticity of isophotes, e = 1-(b/a)

V_HELIO     : Heliocentric radial velocity [km/s]
V_HELIO_ERR : Error on heliocentric radial velocity [km/s]
V_LSR       : Radial velocity relative to the Solar neighbourhood LSR [km/s]
sigma_v     : Central velocity dispersion [km/s]
sigma_v_err : Error on central velocity dispersion [km/s]
c           : King-model central concentration, c = log(r_t/r_c)
r_c         : Core radius [arcmin]
r_h         : Half-light radius [arcmin]
mu_V        : Central surface brightness [V magnitudes/arcsecond^2]
rho_0       : Central luminosity density, log_10(Solar luminosities per cubic parsec)
log(tc)     : Core relaxation time t(r_c), in log_10(years)
log(th)     : Median relaxation time t(r_h), in log_10(years)
```
