import tensorflow as tf
import numpy as np


class TFHollandWindProfile:
    """
    .. |beta|   unicode:: U+003B2 .. GREEK SMALL LETTER BETA

    Holland profile. For `r < rMax`, we reset the wind field to a
    cubic profile to avoid the barotropic instability mentioned in
    Kepert & Wang (2001).

    :param float lat: Latitude of TC centre.
    :param float lon: Longitude of TC centre.
    :param float eP: environmental pressure (hPa).
    :param float cP: centrral pressure of the TC (hPa).
    :param float rMax: Radius to maximum wind (m).
    :param float beta: |beta| parameter.
    :param windSpeedModel: A maximum wind speed model to apply.
    :type  windSpeedModel: :class:`windmodels.WindSpeedModel` instance.

    """

    @staticmethod
    def maximum(dP, beta):
        rho = 1.15
        return tf.sqrt(beta * dP / (np.exp(1) * rho))

    @staticmethod
    def secondDerivative(dP, beta, f, rMax):
        """
        Second derivative of profile at rMax.
        """
        rho = 1.15
        E = np.exp(1)

        d2Vm = ((beta * dP * (-4 * beta ** 3 * dP / rho -
                              (-2 + beta ** 2) * E * (tf.abs(f) * rMax) ** 2)) /
                (E * rho * tf.sqrt((4 * beta * dP) / (E * rho) +
                                   (f * rMax) ** 2) * (4 * beta * dP * rMax ** 2 / rho +
                                                       E * (f * rMax ** 2) ** 2)))

        return d2Vm

    @staticmethod
    def firstDerivative(dP, beta, f, rMax):
        """
        First derivative of profile at rMax
        """
        rho = 1.15
        E = np.exp(1)

        dVm = (-tf.abs(f) / 2 + (E * (f ** 2) * rMax *
                                 tf.sqrt((4 * beta * dP / rho) / E +
                                         (f * rMax) ** 2)) /
               (2 * (4 * beta * dP / rho + E * (f * rMax) ** 2)))
        return dVm

    @staticmethod
    def velocity(dP, beta, f, rMax, R):
        """
        Calculate velocity as a function of radial distance.
        Represents the velocity of teh gradient level vortex.

        :param R: :class:`numpy.ndarray` of distance of grid from
                  the TC centre (metres).

        :returns: Array of gradient level wind speed.
        :rtype: :class:`numpy.ndarray`

        """
        rho = 1.15
        vMax = TFHollandWindProfile.maximum(dP, beta)

        d2Vm = TFHollandWindProfile.secondDerivative(dP, beta, f, rMax)
        dVm = TFHollandWindProfile.firstDerivative(dP, beta, f, rMax)
        aa = ((d2Vm / 2. - (dVm - vMax / rMax) /
               rMax) / rMax)
        bb = (d2Vm - 6. * aa * rMax) / 2.
        cc = dVm - 3. * aa * rMax ** 2 - 2. * bb * rMax
        delta = (rMax / R) ** beta
        edelta = tf.exp(-delta)

        V = (tf.sqrt((dP * beta / rho) *
                     delta * edelta + (R * f / 2.) ** 2) -
             R * tf.abs(f) / 2.)

        icore = tf.cast(R <= rMax, tf.dtypes.float32)
        V = V * (1. - icore) + icore * (R * (R * (R * aa + bb) + cc))
        V = tf.sign(f) * V
        return V

    @staticmethod
    def vorticity(dP, beta, f, rMax, R):
        """
        Calculate the vorticity associated with the (gradient level)
        vortex.

        :param R: :class:`numpy.ndarray` of distance of grid from
                  the TC centre (metres).

        :returns: Array of gradient level (relative) vorticity.
        :rtype: :class:`numpy.ndarray`

        """

        rho = 1.15
        vMax = TFHollandWindProfile.maximum(dP, beta)
        beta = beta
        delta = (rMax / R) ** beta
        edelta = tf.exp(-delta)

        Z = tf.abs(f) + \
            (beta ** 2 * dP * (delta ** 2) * edelta /
             (2 * rho * R) - beta ** 2 * dP * delta * edelta /
             (2 * rho * R) + R * f ** 2 / 4) / \
            tf.sqrt(beta * dP * delta * edelta /
                    rho + (R * f / 2) ** 2) + \
            (tf.sqrt(beta * dP * delta * edelta /
                     rho + (R * f / 2) ** 2)) / R

        # Calculate first and second derivatives at R = Rmax:
        d2Vm = TFHollandWindProfile.secondDerivative(dP, beta, f, rMax)
        dVm = TFHollandWindProfile.firstDerivative(dP, beta, f, rMax)
        aa = ((d2Vm / 2 - (dVm - vMax /
                           rMax) / rMax) / rMax)
        bb = (d2Vm - 6 * aa * rMax) / 2
        cc = dVm - 3 * aa * rMax ** 2 - 2 * bb * rMax

        icore = tf.cast(R <= rMax, tf.dtypes.float32)
        Z = Z * (1.0 - icore) + icore * (R * (R * 4 * aa + 3 * bb) + 2 * cc)
        Z = tf.sign(f) * Z
        return Z


class TFKerpertHolland:

    @staticmethod
    def field(dP, beta, f, rMax, R, lam, vFm, thetaFm, thetaMax=0.):
        """
        :param R: Distance from the storm centre to the grid (km).
        :type  R: :class:`numpy.ndarray`
        :param lam: Direction (geographic bearing, positive clockwise)
                    from storm centre to the grid.
        :type  lam: :class:`numpy.ndarray`
        :param float vFm: Foward speed of the storm (m/s).
        :param float thetaFm: Forward direction of the storm (geographic
                              bearing, positive clockwise, radians).
        :param float thetaMax: Bearing of the location of the maximum
                               wind speed, relative to the direction of
                               motion.

        """

        V = TFHollandWindProfile.velocity(dP, beta, f, rMax, R)
        Z = TFHollandWindProfile.vorticity(dP, beta, f, rMax, R)
        K = 50.  # Diffusivity
        Cd = 0.002  # Constant drag coefficient
        Vm = TFHollandWindProfile.maximum(dP, beta)

        mask = tf.cast(vFm > 0, tf.dtypes.float32) * tf.cast(Vm / vFm < 5., tf.dtypes.float32)

        Umod = mask * vFm * tf.abs(1.25 * (1. - (vFm / Vm)))
        Umod += (1.0 - mask) * vFm

        core = tf.cast(R > 2. * rMax, tf.dtypes.float32)

        Vt1 = Umod * (1 - core)
        Vt2 = core * (Umod * tf.exp(-((R / (2. * rMax)) - 1.) ** 2))
        Vt = Vt1 + Vt2

        al = ((2. * V / R) + f) / (2. * K)
        be = (f + Z) / (2. * K)
        gam = (V / (2. * K * R))

        albe = tf.sqrt(al / be)

        ind = tf.abs(gam) > tf.sqrt(al * be)
        chi = tf.abs((Cd / K) * V / tf.sqrt(tf.sqrt(al * be)))
        eta = tf.abs((Cd / K) * V / tf.sqrt(tf.sqrt(al * be) + tf.abs(gam)))
        psi = tf.abs((Cd / K) * V / tf.sqrt(tf.abs(tf.sqrt(al * be) -
                                                   tf.abs(gam))))

        ind = tf.complex(tf.cast(ind, chi.dtype), np.zeros_like(chi.numpy()))
        eta = tf.complex(eta, np.zeros_like(chi.numpy()))
        albe = tf.complex(albe, np.zeros_like(chi.numpy()))
        psi = tf.complex(psi, np.zeros_like(chi.numpy()))
        chi = tf.complex(chi, np.zeros_like(chi.numpy()))

        lamcomp = tf.complex(lam * tf.sign(f), np.zeros_like(V.numpy()))
        Vtcomp = tf.complex(Vt, np.zeros_like(V.numpy()))
        Vcomp = tf.complex(V, np.zeros_like(V.numpy()))
        i = tf.cast(tf.complex(0., 1.), tf.dtypes.complex64)

        A0 = -(chi * (1 + i * (1 + chi)) * Vcomp) / (2 * chi ** 2 + 3 * chi + 2)
        # Symmetric surface wind component
        u0s = tf.math.real(A0 * albe) * tf.sign(f)
        v0s = tf.math.imag(A0)

        Am = -(psi * (1 + 2 * albe + (1 + i) * (1 + albe) * eta) * Vtcomp) / \
             (albe * ((2 + 2 * i) * (1 + eta * psi) + 3 * psi + 3 * i * eta))
        AmIII = -(psi * (1 + 2 * albe + (1 + i) * (1 + albe) * eta) * Vtcomp) / \
                (albe * ((2 - 2 * i + 3 * (eta + psi) + (2 + 2 * i) *
                          eta * psi)))

        Am = Am * (1.0 - ind) + ind * AmIII

        #         # First asymmetric surface component
        ums = tf.math.real(Am * tf.exp(-i * lamcomp) * albe)
        vms = tf.math.imag(Am * tf.exp(-i * lamcomp)) * tf.sign(f)

        Ap = -(eta * (1 - 2 * albe + (1 + i) * (1 - albe) * psi) * Vtcomp) / \
             (albe * ((2 + 2 * i) * (1 + eta * psi) + 3 * eta + 3 * i * psi))
        ApIII = -(eta * (1 - 2 * albe + (1 - i) * (1 - albe) * psi) * Vtcomp) / \
                (albe * (2 + 2 * i + 3 * (eta + psi) + (2 - 2 * i) *
                         eta * psi))
        Ap = Ap * (1.0 - ind) + ind * ApIII

        # Second asymmetric surface component
        ups = tf.math.real(Ap * tf.exp(i * lamcomp) * albe)
        vps = tf.math.imag(Ap * tf.exp(i * lamcomp)) * tf.sign(f)

        # Total surface wind in (moving coordinate system)
        us = u0s + ups + ums
        vs = v0s + vps + vms + V

        usf = us + Vt * tf.cos(lam - thetaFm)
        vsf = vs - Vt * tf.sin(lam - thetaFm)
        phi = tf.math.atan2(usf, vsf)

        # Surface winds, cartesian coordinates
        Ux = tf.sqrt(usf ** 2. + vsf ** 2.) * tf.sin(phi - lam)
        Vy = tf.sqrt(usf ** 2. + vsf ** 2.) * tf.cos(phi - lam)

        return Ux, Vy


def fcast(x):
    return tf.cast(x, tf.float32)
