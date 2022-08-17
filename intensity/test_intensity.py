import os
import sys
sys.path.append(os.path.expanduser("~/geoscience/repos/tcha"))
from intensity.tc_intensity_analysis import find_dP, Hurricane


# rm hurr.cpython-38-darwin.so;  f2py3 -m hurr -c hurr.f; pytest test_intensity.py


def test_intensity():
    r0 = 480
    sst = 30
    h_a = 83
    lat = 12
    ahm = 86
    sp = 1006
    tend = 1
    ut = 2.8
    hm = 8.6
    gm = 0.19
    dsst = 5

    hurricane = Hurricane()
    vm = 8
    rm = 60
    nn = 20

    for i in range(nn):
        bailed, out = hurricane.pytc_intensity(
            vm, rm, r0, sst, h_a, abs(lat), ahm, sp, tend / nn,
            ut, hm=hm, gm=gm, dsst=dsst,
        )
        pmin, vm, rm = out[0], out[1], out[2]

    hurricane = Hurricane()
    vm_ = 8
    rm = 60
    bailed, out = hurricane.pytc_intensity(
        vm_, rm, r0, sst, h_a, abs(lat), ahm, sp, tend,
        ut, hm=hm, gm=gm, dsst=dsst,
    )
    assert abs(vm - out[1]) <= 0.01
