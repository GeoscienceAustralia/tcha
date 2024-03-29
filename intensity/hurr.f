!     this is adapted from Emanuel's hurricane code - the only apdations have been to wrap
!     the code in a function for interoperation with Python
      subroutine tc_intensity(nrd, time, vm,
     & rm, r0, ts, to, h_a, alat, tshear, vext, tland, surface,
     & hs, om, ut, nr,
     & dt, ro, ahm, pa, cd, cd1, cdcap,
     & cecd, pnu, taur, radmax, tauc, efrac, dpb, hm, dsst, gm, xx,
     & rbs1, rts1, x1, xs1, xm1, mu1, rbs2, rts2, x2, xs2,
     & xm2, mu2, uhmix1, uhmix2, sst1, sst2, ps2, ps3, hmix, init,
     & match, vobs, gconst, shearconst, diagnostic)

      real h_a, meq, mf, mt, mumax, mdmin, vobs, gconst, shearconst
      real time, vm,rm,r0,ts,to,alat,tland,tshear,vext,ut
      real eddytime,rwide,ro,ahm,pa,cd,cd1,cdcap,cecd,pnu,taur
      real radmax,tauc,efrac,dsst,gm,hs,hm,heddy
      real dt,tt,atime
      character*4 om,surface
      character*1 init, match
      real sstr(200), sst1(200),sst2(200)
      real sst3(200)
      real xx(4)

c  ***     dimension arrays of dependent variables   ***
c
      real rbs1(200), rbs2(200), rbs3(200), rts1(200), rts2(200)
      real x1(200), x2(200), x3(200), xs1(200), xs2(200), xs3(200)
      real xm1(200), xm2(200), xm3(200),rts3(200)
      real mu1(200), mu2(200), mu3(200), diagnostic(200)
c
c  ***     dimension ocean variables   ***
c
      real hmix(200), uhmix1(200), uhmix2(200), uhmix3(200)
c
c  ***      dimension various diagnostic quantities     ***
c
      real p(200), ps0(nrd), ps2(200), ps3(200), gb(200), rms2(200)
      real rb1(200), rb2(200), rt1(200)
c
c  ***   dimension viscous and working arrays   ***
c
      real vis(200), vis2(200), xvis(200), dv(200), af(200)
      real q2(200), q3(200), xmvis(200), rmm2(200), epre(200)

      if(hs.lt.0.2.and.surface.eq.'swmp')then
       print*, 'swamp depth must be at least 0.2 meters'
       print*, 'setting depth to 0.2 meters and proceeding'
       hs=max(hs,0.2)
      end if
c
c      *** non-dimensionalize parameters and initial conditions  ***
c
c
c   ***  more external parameters  ***
c
      xmdif=3000.0*float(nr)*0.01
      ric=1.0
      zgrad=100.0
      timed=time
c
      h_a=0.01*h_a
      ahm=0.01*ahm
      es=6.112*exp(17.67*ts/(243.5+ts))
      ea=h_a*es
      qs=0.622*es/pa
      tsa=ts+273.15
      toa=to+273.15
      ef=(ts-to)/tsa
      chi=2.5e6*ef*qs*(1.-h_a)
      schi=sqrt(chi)
      fc=(3.14159/(12.*3600.))*sin(3.14159*alat/180.)
      cd=cd*0.001
      cd1=cd1*1.0e-5
      atha=1000.*log(tsa)+2.5e6*qs*h_a/tsa-qs*h_a*
     &    461.*log(h_a)
      theta_e=exp((atha-287.*log(0.001*pa))*0.001)
      pt=1000.0*(toa/theta_e)**3.484
      delp=0.5*(pa-pt)
      gamma=dpb/delp
      tm=0.85*ts+0.15*to
      esm=6.112*exp(17.67*tm/(243.5+tm))
      qsm=0.622*esm/(pa-0.5*delp)
      tma=tm+273.15
      gratb=(1.+2.5e6*qsm/(287.*tma))/
     &   (1.+2.5e6*2.5e6*qsm/(461.*1005.*tma*tma))
      alength=schi/fc
      atime=287.*tsa*delp/(cd*9.8*pa*schi)
      beta=chi/(287.*tsa)
      q=0.5*h_a/(gratb*(1.-h_a))
      tauc=tauc*3600./atime
      al=1./tauc
      taur=taur*3600./atime
      rad=1./taur
      radmax=radmax/(3600.*24.)
      radmax=radmax*atime*1000.*(ts-to)/(chi*gratb*320.0)
      r0=r0*1000./alength
      rm=rm*1000./alength
      ro=ro*1000./alength
      time=time*3600.*24./atime
      dt=dt/atime
      tland=tland*24.*3600./atime
      tshear=tshear*24.*3600./atime
      vm=vm/schi
      vobs=vobs/schi
      vext1=2.0*vext*atime/alength
      xm0=2.5e6*(ts-to)*qsm*(ahm-1.)/tma
      xm0=xm0/chi
      cdv=cd1*schi/cd
      rhaa=h_a
      gm=gm*0.01
      ut=ut*atime/alength
      amixfac=0.5*1000.*ef*gm*(1.+2.5e6*2.5e6*qsm/(1000.*461.*
     &   tsa*tsa))/chi
      hm=hm*amixfac
        dsst=dsst*2.*amixfac/gm
      amix=(2.*ric*chi/(9.8*3.3e-4*gm))*1.0e-6*(287.*tsa/9.8)**2*
     &   (delp/pa)**2*amixfac**4
      amix=sqrt(amix)
      ni=150
      facsst=0.002*cd*cecd*chi*schi*atime*amixfac/(hs*gm*4160.0)
      xx(4) = 0

c
c   ***   set certain constants    ***
c
      dr=ro/float(nr-2)
      dri=1./dr
      nt=time/dt
      damp=0.1
      dpr=1./gamma
      bfac=1.-dr*sqrt(2./q)
      pbm=1.0e-5
      sixti=1./16.
c
c              *** set actual diffusivities proportional to dr  ***
c
      pnu=pnu*pnu*dr*0.01744
c
c            ****   initialize fields   ***
c

      if (init.eq.'y') then
      do 60 i=2,nr

            r=float(i-1)*dr
            if(r.gt.r0)goto 40
            if(r.gt.sqrt(rm*rm*(1.+2.*vm/rm)))goto 30

            rbs1(i)=(r*r/(1.+2.*vm/rm))
            rbs2(i)=rbs1(i)
            goto 50

30    rbs1(i)=(0.5*r*r-vm*rm/(1.-(rm/r0)**2))/
     &(0.5-vm*rm/(r0*r0-rm*rm))
            rbs2(i)=rbs1(i)
            goto 50

   40       rbs1(i)=r*r
            rbs2(i)=rbs1(i)
   50   continue

            rts1(i)=r*r
            rts2(i)=rts1(i)
            rbs3(i)=rbs2(i)
            rts3(i)=rts2(i)
            x1(i)=-0.05
            x2(i)=x1(i)
            xm1(i)=xm0
            xm2(i)=xm0
            mu1(i)=0.0
            mu2(i)=0.0
            mu3(i)=0.0

       sst1(i)=0.0
       sst2(i)=0.0
       hmix(i)=hm
       uhmix1(i)=sqrt(hmix(i)*hmix(i)*hmix(i)*dsst)/amix
       uhmix2(i)=uhmix1(i)
   60   continue

      hmix(1)=hm
      do 70 i=3,nr
       r=(float(i-2))*dr
       xs1(i)=xs1(i-1)+r*dr*0.5*(1.-r*r/rbs1(i-1))
   70 continue
        mu2(1)=0.0
      sst2(1)=0.0
      sst1(1)=0.0
      do i=2,nr
        xs1(i)=xs1(i)-xs1(nr)
        xs2(i)=xs1(i)
        ps2(i)=0.0
        ps3(i)=0.0
      end do
      ps3(1)=0.0
      ps2(1)=0.0
c
c
      end if

      do 75 i=2,nr
        rb1(i)=sqrt(rbs1(i))
!       ps2(i)=0
        rb2(i)=rb1(i)
        rt1(i)=sqrt(rts1(i))
        xvis(i)=0.0
        xmvis(i)=0.0
!       ps3(i)=0.0
        p(i)=0.0
c	 if((rb2(i)*0.001*alength).lt.100.0)then
c	  xm1(i)=xs1(i)
c	  xm2(i)=xm1(i)
c       end if
   75 continue

      ps0(1)=0.0
      vis(1)=0.0
      rmm2(1)=0.0

      gb(1)=0.0
c
c             ***  set time looping parameters and begin time loop  ***
c
      tt=-dt
      tt1=0.0
      ntt=0
      dt1=dt
c
c           ***  program returns to 77 after each time step  ***
c

   77 continue
      tt=tt+dt

      if(tt.gt.time)goto 705
      ntt=ntt+1
c
c            *** set boundary values for and advection terms ***
c
      rbs1(1)=0.0
      rts1(1)=0.0
      rbs2(1)=0.0
      rbs3(1)=0.0
      rts2(1)=0.0
      rts3(1)=0.0
      rms2(1)=0.0
      rb1(1)=0.0
      rb2(1)=0.0
      rt1(1)=0.0
      rts2(nr)=rts2(nr-1)+dr*(dr+2.*ro)
      rts1(nr)=rts1(nr-1)+dr*(dr+2.*ro)
      rbs2(nr)=rbs2(nr-1)+dr*(dr+2.*ro)
      rbs1(nr)=rbs1(nr-1)+dr*(dr+2.*ro)
      rb1(nr)=sqrt(rbs1(nr))
      rb2(nr)=sqrt(rbs2(nr))
      rt1(nr)=sqrt(rts1(nr))
      x1(1)=x1(2)
      xs2(1)=xs2(2)
      xs1(1)=xs1(2)
      x2(1)=x2(2)
      xm2(1)=xm2(2)
      xm1(1)=xm1(2)
      mu2(1)=mu2(2)
      sst2(1)=sst2(2)
      sst2(nr)=0.0
      mu1(nr-1)=0.0
      mu2(nr-1)=0.0
      x1(nr)=x1(nr-1)
      x2(nr)=x2(nr-1)
      xs1(nr)=xs1(nr-1)
      xs2(nr)=xs2(nr-1)
      xm2(nr)=xm2(nr-1)
      xm1(nr)=xm1(nr-1)
      xvis(nr)=xvis(nr-1)
      xmvis(nr)=xmvis(nr-1)
      uhmix1(1)=uhmix1(2)
      uhmix2(1)=uhmix2(2)
      tfac=hm
      uhmix1(nr)=sqrt(tfac*tfac*tfac*dsst)/amix
      uhmix2(nr)=uhmix1(nr)
c
      vmax=0.0
      idx = -1
      do i=2,nr-1
       r=(float(i-1))*dr
       v=0.5*(r*r-rbs2(i))/sqrt(rbs2(i))
       if (v > vmax) idx = i
       vmax=max(v,vmax)
      end do
c      if (match.EQ.'y') print *, "idx:", idx, vmax * schi,schi,alength
      vext=0.0
      if(tt.ge.tshear)vext=0.02*vmax*vmax*vext1*vext1
c
c           ***  in this loop we calculate the integrated cumulus
c                mass flux, the total pbl streamfunction, and
c                the diffusive terms   ***
c
      do 108 i=2,nr-1
       r=(float(i-1))*dr
       rp=r+0.5*dr
       r2=r+dr
       r1=r-dr
       r10=r1-dr
c
c            *** calculate integrated cumulus mass flux ***
c
c            ***        precipitation efficiency        ***
c
       denom=max(0.001,(x1(i)-xm0))
       fac=(xm1(i)-xm0)/denom
       fac=max(fac,0.0)
       fac=min(fac,1.0)
       epre(i)=fac
c
       mt=mu2(i)*epre(i)
       gb(i)=gb(i-1)-mt*(rbs2(i)-rbs2(i-1))
c
c            *** calculate ekman flow ps0(i) ***
c
       ps0(i)=0.25*((rb1(i+1)-rb1(i-1))
     1     *(r*r-rbs1(i))*abs(r*r-rbs1(i)))/(r*dr)
       vrel=0.5*(r*r-rbs1(i))/rb1(i)
       vabs=abs(vrel)
       cdfac=1.0+cdv*vabs
       cdfac=min(cdfac,cdcap)
c
      if(tt.ge.tland)then
       if(surface.eq.'swmp')then
        cdfac=1.0
       else if(surface.eq.'pln')then
        cdfac=1.5
       else if(surface.eq.'hill')then
        cdfac=2.5
       else if(surface.eq.'mtn')then
        cdfac=4.0
       end if
      end if
c
       ps0(i)=ps0(i)*cdfac
c
c            ***   calculate viscous term for layer 1  ***
c
       c2=(r2*r2/rbs1(i+1)-r*r/rbs1(i))/(rb1(i+1)-rb1(i))
       rm2=0.5*(rb1(i+1)+rb1(i))
       rm2=rm2*rm2
       rm2=rm2*rm2
       if(i.eq.2)then
         c1=(r2*r2/rbs1(i+1)-r*r/rbs1(i))/rb1(i)
       else
         c1=(r*r/rbs1(i)-r1*r1/rbs1(i-1))/(rb1(i)-rb1(i-1))
       end if
       rm1=0.5*(rb1(i)+rb1(i-1))
       rm1=rm1*rm1
       rm1=rm1*rm1
       vis(i)=-(pnu/(r*dr))*(rm2*c2*abs(c2)-rm1*c1*abs(c1))
c
c            ***   calculate viscous term for layer 2  ***
c
       c2=(r2*r2/rts1(i+1)-r*r/rts1(i))/(rt1(i+1)-rt1(i))
       rm2=0.5*(rt1(i+1)+rt1(i))
       rm2=rm2*rm2
       rm2=rm2*rm2
       if(i.eq.2)then
         c1=(r2*r2/rts1(i+1)-r*r/rts1(i))/rt1(i)
       else
         c1=(r*r/rts1(i)-r1*r1/rts1(i-1))/(rt1(i)-rt1(i-1))
       end if
       rm1=0.5*(rt1(i)+rt1(i-1))
       rm1=rm1*rm1
       rm1=rm1*rm1
       vis2(i)=-(pnu/(r*dr))*(rm2*c2*abs(c2)-rm1*c1*abs(c1))
c
c         ***  calculate diffusion term in thermodynamic equation  ***
c
       if(i.eq.2)then
        c2=2.*rbs1(i)*abs(r2*r2/rbs1(i+1)-r*r/rbs1(i))/rbs1(i+1)
        bs=c2
        c1=0.0
       else
        c2=rbs1(i)*abs(r2*r2/rbs1(i+1)-r1*r1/rbs1(i-1))/
     1    ((rb1(i+1)-rb1(i-1))*(rb1(i+1)-rb1(i-1)))
       end if
       if(i.eq.3)then
        c1=bs
       else if(i.gt.3)then
        c1=rbs1(i-1)*abs(r*r/rbs1(i)-r10*r10/rbs1(i-2))/
     1    ((rb1(i)-rb1(i-2))*(rb1(i)-rb1(i-2)))
       end if
       xvis(i)=4.*pnu*(c2*(xs1(i+1)-xs1(i))-c1*(xs1(i)-
     1     xs1(i-1)))/(rbs1(i)-rbs1(i-1))
c
c          *** calculate r**2 at the middle level ***
c
       rms2(i)=rbs1(i)/(0.7+rbs1(i)*0.3/rts1(i))
       rmm2(i)=sqrt(rms2(i))
c
  108   continue
c
c          ***  various boundary values   ***
c
      rms2(nr)=rms2(nr-1)+dr*(dr+2.*ro)
      rmm2(nr)=sqrt(rms2(nr))
      gb(nr)=gb(nr-1)
      ps0(nr)=ps0(nr-1)*bfac
      epre(nr)=epre(nr-1)
      epre(1)=epre(2)
c
c   viscous term for middle layer
c
        do 109 i=2,nr-1
         r=(float(i-1))*dr
         rp=r+0.5*dr
         r2=r+dr
         r1=r-dr
         r10=r1-dr
         if(i.eq.2)then
          c2=2.*rms2(i)*abs(r2*r2/rms2(i+1)-r*r/rms2(i))/rms2(i+1)
          c2b=rmm2(i)
          bs=c2
          bsb=c2b
          c1=0.0
          c1b=0.0
         else
          c2=rms2(i)*abs(r2*r2/rms2(i+1)-r1*r1/rms2(i-1))/
     1    ((rmm2(i+1)-rmm2(i-1))*(rmm2(i+1)-rmm2(i-1)))
          c2b=rmm2(i)
         end if
         if(i.eq.3)then
          c1=bs
          c1b=bsb
         else if(i.gt.3)then
          c1=rms2(i-1)*abs(r*r/rms2(i)-r10*r10/rms2(i-2))/
     1    ((rmm2(i)-rmm2(i-2))*(rmm2(i)-rmm2(i-2)))
          c1b=rmm2(i-1)
         end if
       c1=c1b*4.*xmdif*pnu
       c2=c2b*4.*xmdif*pnu
c	 c1=2.*c1*pnu
c	 c2=2.*c2*pnu
       xmvis(i)=(c2*(xm1(i+1)-xm1(i))-c1*(xm1(i)-
     1     xm1(i-1)))/(rms2(i)-rms2(i-1))
  109 continue
c
c         *** calculate arrays for solving elliptic equation for psi ***
c
      do 150 i=2,nr-1
       r=(float(i-1))*dr
       rf2=rad*xs1(i+1)
       rf2=min(rf2,radmax)
       rf1=rad*xs1(i)
       rf1=min(rf1,radmax)
       gf1=r*r/rts2(i)
       gf2=r*r/rbs2(i)
       gf=gf1*gf1+gf2*gf2
       df=0.5*gf*dr/r
       q3(i)=q/(rms2(i+1)-rms2(i))
       q2(i)=q/(rms2(i)-rms2(i-1))
       rat=(rbs2(i)/rts2(i))
       rat=rat*rat
       pf=(ps0(i)+vis(i))/(1.+rat)+gb(i)
       pf=pf-vis2(i)*rat/(1.+rat)
       dv(i)=df*pf+rf2-rf1-xvis(i+1)+xvis(i)
       af(i)=1./(q2(i)+q3(i)+df)
  150 continue
c
c           ***  solve elliptic equation for psi  ***
c
      do 360 k=1,1000
       kp=k
       pmax=0.0
       ps3(nr)=ps3(nr-1)*bfac+gb(nr-1)*(1.-bfac)
       do 320 i=nr-1,2,-1
        r=(float(i-1))*dr
        ps3(i)=af(i)*(dv(i)+q3(i)*ps3(i+1)+q2(i)*ps2(i-1))
        a=ps3(i)-ps2(i)
        pmax=max(pmax,abs(a))
  320    continue
       do 340 i=2,nr-1
        ps2(i)=ps3(i)
  340    continue
       if(pmax.lt.pbm)goto 370
  360   continue
  370   continue
c
c            ***  solve surface pressure equation  ***
c
      p(2)=0.0
      pav=0.0
      do 200 i=3,nr-1
       r=(float(i-1))*dr
       rmi=r-dr
       r4=r*r*r*r
       rmi2=rmi*rmi
       rmi3=rmi*rmi2
       rmi4=rmi3*rmi
       rmim4=(rmi-dr)*(rmi-dr)
       rmim4=rmim4*rmim4
       fm1=xs2(2)+sixti*(rmi4/rbs2(2)+rbs2(2))
       if(i.gt.3)fm1=sixti*((rmi4/rbs2(i-1))+rbs2(i-1)+(
     1    rmim4/rbs2(i-2))+rbs2(i-2))+xs2(i-1)
       fm2=sixti*((r4/rbs2(i))+rbs2(i)+(rmi4/rbs2(i-1))
     1     +rbs2(i-1))+xs2(i)
       p(i)=p(i-1)+0.5*dr*rmi3/rts2(i-1)+fm1-fm2
       pav=pav+p(i)*(rbs2(i)-rbs2(i-1))
  200 continue
      pav=pav/rbs2(nr-1)
c
c           ***  rectify pressures  ***
c
      do 250 i=2,nr-1
        p(i)=p(i)-pav
  250 continue
      p(1)=p(2)
      p(nr)=p(nr-1)
c
c            ***  in this loop the right-hand sides of the
c                 time-dependent equations are calculated and the
c                 quantities are then incremented in time ***
c
      do 100 i=2,nr-1
      r=(float(i-1))*dr
      r2=r+dr
      r1=r-dr
c
c            ***  calculate forcing of rt and rb ***
c
      rtf=ps2(i)-gb(i)+vis2(i)
      rbf=ps0(i)+vis(i)-ps2(i)+gb(i)
c
c            ***  calculate vertical advection terms in s* eqn   ***
c
      vadv=-q2(i)*(ps2(i)-ps2(i-1))
      diagnostic(i) = vadv
c
c            *** calculate v at x points and pbl advections  ***
c
      vb2=0.5*(r*r-rbs2(i))/rb2(i)
      ub1=ps0(i)/((rb2(i+1)-rb2(i-1)))
      if(i.eq.2)then
       ub1=max(ub1,0.0)
       ub2=0.0
       vb1=0.0
      else
       vb1=0.5*((r-dr)*(r-dr)-rbs2(i-1))/rb2(i-1)
       ub2=ps0(i-1)/(rb2(i)-rb2(i-2))
      end if
c
c  ***  here we form a weighted average of centered and upstream  ***
c  ***      differences to provide some smoothing    ***
c
      ubi=2.*min(ub2,0.0)
      ubo=2.*max(ub1,0.0)
      ubi=0.8*ub2+0.2*ubi
      ubo=0.8*ub1+0.2*ubo
      adv=(ubo*(x2(i+1)-x2(i))+ubi*(x2(i)-x2(i-1)))/
     1   (rb2(i)+rb2(i-1))
      advm=(ubo*(mu2(i+1)-mu2(i))+ubi*(mu2(i)-mu2(i-1)))/
     1   (rb2(i)+rb2(i-1))
c
c        *** calculate vertical velocity at top of ekman layer ***
c
      w0=(ps0(i)-ps0(i-1))/(rbs2(i)-rbs2(i-1))
c
c  ***    calculate downdraft and vertical velocity in between clouds   ***
c
      amd=-mu2(i)*(1.-epre(i))
      w0d=w0-mu2(i)-amd
c      diagnostic(i) = w0
c
c  *** calculate surface fluxes, radiation and equilibrium mass flux   ***
c  ***          include dissipative heating term here                  ***
c
      vabs=0.5*abs(vb1+vb2)
      cdfac=1.0+cdv*vabs
      cdfac=min(cdfac,cdcap)
      ckfac=cdfac
      if(tt.ge.tland)then
       if(surface.eq.'swmp')then
        ckfac=1.0
       else
        ckfac=0.0
       end if
       cdfac=1.0
       if(surface.eq.'pln')then
        cdfac=1.5
       else if(surface.eq.'hill')then
        cdfac=2.5
       else if(surface.eq.'mtn')then
        cdfac=4.0
       end if
      end if
c
c   ***  calculate forcing of ocean temperature  ***
c
      sstr(i)=0.0
      seaf=0.0
      sstf=0.0
      if(om.eq.'y'.or.om.eq.'y'.and.tt.lt.tland)then
       ao1=(uhmix2(i+1)-uhmix2(i))/(rb2(i+1)-rb2(i))
       ao2=(uhmix2(i)-uhmix2(i-1))/(rb2(i)-rb2(i-1))
       advom=(ut+0.5*rbf/rb2(i))*(0.8*ao1+0.2*ao2)
c	 seaf=advom+cdfac*vten*vten+spmfac*vten*spfunc
       seaf=advom+cdfac*vabs*vabs
       ahmlocal1=hm
       ahmlocal2=hm
       amixd1=max(hmix(i),0.0001)
       amixd2=max(hmix(2),0.0001)
       xsurm1=-((hmix(i)-ahmlocal1)**2)/amixd1
       xsurm2=-((hmix(2)-ahmlocal2)**2)/amixd2
       xsurm1=xsurm1+dsst*(ahmlocal1-hmix(i))/amixd1
       xsurm2=xsurm2+dsst*(ahmlocal2-hmix(2))/amixd2
       sstr(i)=0.5*(xsurm1+xsurm2)
      end if
c
      if(tt.ge.tland.and.surface.eq.'swmp')then
       sstr(i)=sst2(i)
       xsur=1.0+sstr(i)-ef*p(i)+(exp(-beta*p(i))-1.)/(1.-h_a)
       as1=(sst2(i+1)-sst2(i))/(rb2(i+1)-rb2(i))
       as2=(sst2(i)-sst2(i-1))/(rb2(i)-rb2(i-1))
       advos=(ut+0.5*rbf/rb2(i))*(0.8*as1+0.2*as2)
       sstf=-facsst*ckfac*vabs*(xsur-x1(i))+advos
      else
       xsur=1.0+sstr(i)-ef*p(i)+(exp(-beta*p(i))-1.)/(1.-h_a)
      end if
c
      flux=ckfac*vabs*(xsur-x1(i))*cecd
      flux=flux+ef*cdfac*vabs*vabs*vabs
c
      rad1=rad
      if((rad1*xs1(i)).gt.radmax)rad1=radmax/xs1(i)
      if(w0d.le.0.0)then
       achim=max(0.001,(x1(i)-xm1(i)))
       meq=w0+(adv+flux-gamma*(vadv+xvis(i)-rad1*xs1(i)))/
     1    achim
      else
       atemp=(x1(i)-xm1(i))*(1.-epre(i))
       achim=max(0.001,atemp)
       meq=(adv+flux-gamma*(vadv+xvis(i)-rad1*xs1(i)))/
     1  achim
      end if
c
c  ***  limit maximum value of m   ***
c
      meq=min(meq,400.0)
c
c  ***    if conditionally stable, set meq to 0    ***
c
      if(x2(i).lt.(xs2(i)-0.0001))meq=0.0
c
c        *** calculate forcing of mass flux  ***
c
      mf=0.1*advm+al*(meq-mu1(i))
c
c        *** calculate forcing of entropy eqns ***
c
      wxm=max(0.0,w0d)
c	xmf=(efrac*mu2(i)+wxm)*(x1(i)-xm1(i))-(xs1(i)-xm1(i))*
c     1   (ps2(i)-ps2(i-1))/(rms2(i)-rms2(i-1))
      xmf=efrac*(mu2(i)+wxm)*(x1(i)-xm1(i))
      xmf=xmf+xmvis(i)-rad1*xs1(i)*(0.75*gratb+0.25)
      xmf=xmf-shearconst*vext*(xm1(i)-xm0)
      if (match.eq.'y') then
        xmf = xmf + gconst * (vobs - vmax) * (xm1(i)-xm0)
      end if
      amc=min(0.0,w0d)
      xf=adv+flux+(amd+amc)*(x1(i)-xm1(i))
      xf=xf*dpr
      xsf=vadv+xvis(i)-rad1*xs1(i)

c
c           ***  increment in time   ***
c
      rbs3(i)=rbs1(i)+2.*dt*rbf
      rts3(i)=rts1(i)+2.*dt*rtf
      xs3(i)=xs1(i)+2.*dt*xsf
      x3(i)=x1(i)+2.*dt*xf
      x3(i)=min(x3(i),xs3(i),xsur)
      xm3(i)=xm1(i)+2.*dt*xmf
      xm3(i)=max(xm3(i),min(-1.0,xm0))
      xm3(i)=min(xm3(i),x3(i))
      mu3(i)=mu1(i)+2.*dt*mf
      mu3(i)=max(mu3(i),0.0)
      uhmix3(i)=uhmix1(i)+2.*dt*seaf
      uhmix3(i)=max(uhmix3(i),0.0)
      sst3(i)=sst1(i)+2.*dt*sstf
c
c  ***    bail out if r**2 is negative   ***

      if (rbs3(i).lt.0.0) then
!     print *, "Aborting! Negative radius"
!     print *, rbs3(i)
!     print *, rbs1(i)
!     print *, rbf
!     print *, i
!     print *, xm1(i)-xm0
!     print *, ps0(i), vis(i), ps2(i), gb(i)
      xx(4) = 1
      goto 710
      end if
  100 end do

      sstr(nr)=sstr(nr-1)
      sstr(1)=sstr(2)
      sst3(1)=sst3(2)
      sst3(nr)=0.0

      IF(VMAX.GT.1.5)DT=0.5*DT1
      IF(VMAX.GT.2.0)DT=0.2*DT1
c
c
c            ***  advance variables with time smoother ***
c
      do 600 i=2,nr-1
       rbs1(i)=rbs2(i)+damp*(rbs1(i)+rbs3(i)-2.*rbs2(i))
       rts1(i)=rts2(i)+damp*(rts1(i)+rts3(i)-2.*rts2(i))
       rb1(i)=sqrt(rbs1(i))
       rt1(i)=sqrt(rts1(i))
       x1(i)=x2(i)+damp*(x1(i)+x3(i)-2.*x2(i))
       xs1(i)=xs2(i)+damp*(xs1(i)+xs3(i)-2.*xs2(i))
       xm1(i)=xm2(i)+damp*(xm1(i)+xm3(i)-2.*xm2(i))
       mu1(i)=mu2(i)+damp*(mu1(i)+mu3(i)-2.*mu2(i))
       mu1(i)=max(mu1(i),0.0)
       uhmix1(i)=uhmix2(i)+damp*(uhmix1(i)+uhmix3(i)-2.*uhmix2(i))
       sst1(i)=sst2(i)+damp*(sst1(i)+sst3(i)-2.*sst2(i))
c
       rbs2(i)=rbs3(i)
       rts2(i)=rts3(i)
       rb2(i)=sqrt(rbs2(i))
       x2(i)=x3(i)
       xs2(i)=xs3(i)
       xm2(i)=xm3(i)
       mu2(i)=mu3(i)
       sst2(i)=sst3(i)
       uhmix2(i)=uhmix3(i)
       if(om.eq.'y'.or.om.eq.'y')then
         ahm0=hm
         ahm02=ahm0*(ahm0-dsst)
         hmix(i)=sqrt(0.5*ahm02+sqrt(0.25*ahm02*ahm02+amix*amix*
     1   uhmix2(i)*uhmix2(i)))
       end if
  600 continue
c
  700 goto 77
  705 continue
c
  710 continue
      pmin = pa * exp(beta * p(1))
      vmax = 0.0
      rmax = 0.0
      do i=1, nr
            pmin = min(pmin, pa * exp(beta * p(i)))
            r=(float(i-1))*dr
        v=0.5*(r*r-rbs2(i))/rb2(i)

        if (v > vmax) then
            vmax = v
            rmax = RB2(I)
         end if

      end do

      xx(1) = pmin
      xx(2) = vmax * schi
      xx(3) = rmax * schi * 0.001 / FC
!     print *, "Actual sim time: ", (tt-dt)*atime / (3600*24), ntt
!      print *, "pmin: ", pmin

      end


      SUBROUTINE THEORY(CKCD,H,TS,TO,PA, out)
C
C       This subroutine calculates the theoretical maximum wind
C         speed and minimum central pressure.
C
       REAL H, LV, PA, PC, PM, out(1)
       DELTAT=0.0
       LV=2.5E6
       RD=287.0
       RV=461.0
       IFAIL=0
C
C      AN is the assumed power dependence of v on radius inside the
C        radius of maximum winds (i.e. v~r**an) used to calculate PC
C
       AN=1.0
C
        ES=6.112*EXP(17.67*TS/(243.5+TS))
        EP=(TS-TO)/(TO+273.15)
        COEF1=EP*LV*ES/(RV*(TS+273.15))
        COEF2=0.5*CKCD*(1.-H)*(EP+1.)
        COEF3=0.5*CKCD*EP*1000.*DELTAT/(RD*(TS+273.15)*(1.-EP*H))
        PM=PA
        N=0
   10   CONTINUE
        N=N+1
        PG=PA*EXP(-COEF1*(COEF2/PM+H*(1./PM-1./PA))-COEF3)
        IF(ABS(PG-PM).LT.0.1)THEN
         PM=0.5*(PM+PG)
         VM=SQRT(EP*CKCD*(LV*0.622*ES*(1.-H)/PM+1000.*DELTAT))
         GOTO 20
        END IF
        IF(N.GT.1000.OR.PG.LE.1.0)THEN
          IFAIL=1
          GOTO 20
        END IF
        PM=0.5*(PM+PG)
        GOTO 10
   20   CONTINUE
	IF(IFAIL.EQ.1)THEN
	 PC=PA
	 VM=0.0
	ELSE
         PC=PM*EXP(-VM*VM/(2.*AN*RD*(TS+273.15)))
	END IF
C
      out(1) = VM

      RETURN
      END