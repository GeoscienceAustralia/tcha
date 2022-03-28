import tcint

def pytc_intensity(vm, rm, r0, ts, h_a, alat, nr, dt, ahm, pa, hm=30.0, dsst=0.6, gm=8.0):
    """

    Parameters
    ----------
    vm : maximum wind speed (m/s)
    rm : radius of maximum winds (km)
    r0 : radius of zero wind speed (km)
    ts : sea surface temperature (C)
    h_a : undisturbed humidity of environmental air near the surface (percent)
    alat : latitude (degrees)
    nr : number of radial nodes
    dt : time step (s)
    ahm : undisturbed relative humidity of undisturbed lower and middle troposphere
    pa : undisturbed sea surface pressure (hPa)
    hm : undisturbed ocean mixed layer depth (m)
    dsst : temperature jump at base of mixed layer (C)
    gm : Sub mixed layer ocean thermal stratification (C/100m)

    Returns
    -------

    """
    nrd = 200
    time = 10  # days
    vdisp = 's'  # plotting param
    dtg = 3  # plotting param
    rog = 0  # plotting param
    rst = 'n'   # plotting param
    to = -70  # environmental temp at top of tc Celsius
    tshear = 200  # time until shear days
    vext = 0  # wind shear m/s
    tland = 200  # time until land
    surface = 'pln'  # type of land
    hs = 0  # swamp depth
    om = 'y'  # ocean mixing on
    ut = 7  # speed over ocean m/s
    eddytime = 200  # time until storm reaches eddy
    heddy = 30  # peak ocean eddy m
    rwide = 400  # radius km
    dim = 'y'  # dimensional output
    fmt = 'tab'  # tabs
    ro = 1200  # radius of model boundary km
    cd = 0.8  # drag coefficient
    cd1 = 4  # drag coefficient rate of change w/ windspeed
    cdcap = 3  # max drag
    cecd = 1  # ratio of drag coefficients
    pnu = 0.03  # turbulent mixing length
    taur = 8  # radiative relaxation time scale
    radmax = 1.5  # max rad cooling rate
    tauc = 2  # convective relation time scale
    efrac = 0.5  # fraction of convective entropy detrained into lower
    dpb = 50  # boundary layer depth

    return tcint.tc_intensity(
        nrd, time, vdisp, dtg, rog, rst, vm,
        rm, r0, ts, to, h_a, alat, tshear, vext, tland, surface,
        hs, om, ut, eddytime, heddy, rwide, dim, fmt, nr,
        dt, ro, ahm, pa, cd, cd1, cdcap,
        cecd, pnu, taur, radmax, tauc, efrac, dpb, hm, dsst, gm
    )


if __name__ == "__main__":
    import time
    t0  = time.time()
    pmin = pytc_intensity(15, 80, 350, 27, 80, 20, nr=50, dt=20, ahm=45, pa=1005, hm=30)
    print(pmin)
    print("time: ", time.time() - t0, "s")
