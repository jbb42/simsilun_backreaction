      program simsilun
! code: SIMplified SILent UNiverse, https://bitbucket.org/bolejko/simsilun
! author: Krzysztof Bolejko 
! disclaimer: There is NO warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
! papers: arXiv:1707.01800, arXiv:1708.09143
! licence: GNU General Public License version 3 or any later version.
      implicit none
	integer I,Ii
	integer, parameter :: Nx = 4  ! number of variables evolved with the Silent Universe
	double precision X(Nx)
! X(1) = density
! X(2) = expansion
! X(3) = shear
! X(4) = Weyl
!----------UNITS:-------------
! density and Weyl are x 8pi G/c^2 
! expansion and shear are x 1/c


	integer, parameter :: Ni = 64*64!*64 !2000  ! dimension of the initial data vector - for single value Ni = 1
        double precision Din(Ni),dini ! initial density contrast 
        double precision Rout(Ni), Reds(Ni) ! final density in Silent Universe and within linearly perturbed Einstein-de Sitter model

        double precision InD(10) ! initial data and the final time instant
        double precision cpar(30)  ! vector with cosmological parameters

        character(len=100) :: arg
        double precision :: z_i, z_f, H_0

        ! Read parameters from command line
        call get_command_argument(1, arg)
        read(arg,*) z_i

        call get_command_argument(2, arg)
        read(arg,*) z_f

        call get_command_argument(3, arg)
        read(arg,*) H_0

        print *, "Running with z_i=", z_i, " z_f=", z_f, " H_0=", H_0

! cosmological parameters
	call get_parameters(cpar, H_0)


! load initial data: density contrast vector deli(Ni) and other data in InD(10)
	call initial_data(cpar,InD,Ni,Din,z_i,z_f,H_0)


	Rout = 0.0d0
	Reds = 0.0d0
! calculate the evolution of X(Nx) -> then -> write density to Xout
!$OMP PARALLEL DO PRIVATE(I,dini,X),SHARED(InD,Din,Rout,Reds)
	do I=1,Ni
		dini = Din(I)
		call silent_evolution(InD,dini,Nx,X)
	Rout(I) = (X(1)/InD(5)) 
	Reds(I) = Din(I)*(InD(2)/InD(5))**(1.0/3.0) +1.0
	enddo
!$OMP END PARALLEL DO


! output: initial density contrast ($1) vs final density in the Silent Universe ($2) and Einstein-de Sitter model ($3)
	open(21,file='density')
	do I=1,Ni
  	   write(21,*) Din(I),Rout(I),Reds(I)
	enddo

      end

!=====================================================

	subroutine initial_data(cpar,InD,Ni,Din,z_i,z_f,H_0)
	implicit none
	integer I,Ni       
	double precision InD(10), Din(Ni)
        double precision cpar(30)
        double precision zo,zz,zf,cto,ctf
        double precision, intent(in) :: z_i, z_f, H_0

! InD(1) = initial time instant
! InD(2) = background's density
! InD(3) = background's expansion rate
! InD(4) = final time instant
! InD(5) = final background's density
! InD(6) = cosmological constant
! InD(7) = virialisation
! InD(8) = time step


! initial values: redshift, time instant, density, and expansion rate (the LCDM model assumed)
	zo = z_i!80.0!1090.0d0
	zz = (zo+1.0d0)
	call timelcdm(zo,cto,H_0)
	InD(1) = cto
	InD(2) = cpar(5)*(zz**3)
	InD(3) = 3.0d0*cpar(2)*dsqrt(cpar(3)*(zz**3) + cpar(4))
  
! final time instants
	zf = z_f!0.01!0.0
        call timelcdm(zf,ctf,H_0)
	InD(4) = ctf
	zz = (zf+1.0d0)
	InD(5) = cpar(5)*(zz**3)


! initial vector with density contrasts
!Generate a simple example of initial conditions. 
!Modify this to read in a more realistic set of initial conditions, e.g. from the Millenium simulation initial conditions as in arXiv:1708.09143
!	do I=1,Ni
!	Din(I) = -0.00095+0.000001*I
!	enddo

        open(unit=10, file="grid", status="old", action="read")

            do i = 1, Ni
                read(10, *) Din(i)
                Din(i) = Din(i)-1.0
        end do

        close(10)
! other parameters
	InD(6) = cpar(7)
	InD(7) = cpar(10)
	InD(8) = cpar(11)

	end


!=====================================================

	subroutine silent_evolution(InD,dini,Nx,X)
	implicit none
	integer I,J, Nx,Nf,Nq
	integer option, virialisation
        double precision, intent(in) :: InD(10), dini
	double precision X(Nx),Xi(Nx),Xii(Nx),V(Nx),RK(Nx,4)
	double precision xp,xp1,xp2,xp3,yp(Nx),yp1(Nx),yp2(Nx),yp3(Nx)
	double precision lb,tevo,dt,cti,cto,ctf
        logical collapse


! parameters
	lb = InD(6)
	virialisation = int(InD(7))
	option = int(InD(8))
        collapse = .false.
	
! time of integration, and other time instants:
	cto = InD(1)
	ctf = InD(4)
        tevo = ctf - cto
	cti = cto
	xp1 = cto
	xp2 = cto
	xp3 = cto

! initial conditions
  	X(1) = InD(2)*(1.0d0 + dini        )
	X(2) = InD(3)*(1.0d0 -(dini/3.0d0) )
	X(3) =  (dini/9.0d0)*InD(3) 
	X(4) = -(dini/6.0d0)*InD(2) 
	call get_V(Nx,X,lb,V)
        Xi = X
	Xii= X

! integration steps (see also get_parameters for options 1 and 2)
	if(option==1) then
        dt = dabs(1d-3/(X(2) + 0.33*X(3)))
	Nf = 1000*int(1000.0*tevo*(X(2) + 0.33*X(3)))
	endif
	if(option==2) then
	Nf = 350000
	dt = tevo/(1.0d0*Nf)
	endif
	if(option/=1 .and. option/=2) then
	print *, 'please specify the *option* for the time step'
	print *, '---calculations are being aborted---'
	stop
	endif
	Nq = 0
 101  continue


! evolution
      do I=1,Nf

	if(option==1) dt = dabs(1d-3/(X(2) + X(3)/3.0d0))
	if(option==2) dt = tevo/(1.0d0*Nf)

        cti = cti + dt
	     
	call get_V(Nx,X,lb,V)
                  do J=1,Nx
                  RK(J,1) = dt*V(J)
                  X(J) = Xi(J) + 0.5*RK(J,1) 
                  enddo
	call get_V(Nx,X,lb,V)
                  do J=1,Nx
                  RK(J,2) = dt*V(J)
                  X(J) = Xi(J) + 0.5*RK(J,2) 
                  enddo
	call get_V(Nx,X,lb,V)
                  do J=1,Nx
                  RK(J,3) = dt*V(J)
                  X(J) = Xi(J) + RK(J,3) 
                  enddo
	call get_V(Nx,X,lb,V)
              do J=1,Nx
              RK(J,4) = dt*V(J)
              X(J)=Xi(J)+(RK(J,1)+2.0*(RK(J,2)+RK(J,3))+RK(J,4))/6.0d0
              enddo

! check for the collapse and apply virialisation if necessary
        if(X(2)<=0.0d0) collapse = .true.
	if(collapse) then

		if(virialisation==1) then
                X(2) = 0.0d0
		goto 102
		endif

		if(virialisation==2) then
 	        if(isnan(X(1)) .or. isnan(X(2))) then
		  X = Xii
                  X(2) = 0.0d0
	          goto 102
		  endif
		endif

		if(virialisation==3) then
		X(1) = (InD(5)*60.0d0)
		X(2) = 0.0d0
		X(3) = 0.0d0
		X(4) = 0.0d0
	        goto 102
		endif
	
	endif

! due to implementation of the dynamical step, the time integration
! will overshoot the final instant, hence the Lagrange Interpolation:
	xp1 = xp2
	xp2 = xp3
	xp3 = cti 
	xp  = ctf
        if(cti==ctf) goto 102
        if(cti>ctf) then
  	   yp1 = Xii
	   yp2 = Xi
	   yp3 = X
	   X = 0.0d0
	   X = X + yp1*((xp - xp2)/(xp1-xp2))*((xp - xp3)/(xp1-xp3))
	   X = X + yp2*((xp - xp1)/(xp2-xp1))*((xp - xp3)/(xp2-xp3))
	   X = X + yp3*((xp - xp1)/(xp3-xp1))*((xp - xp2)/(xp3-xp2))
	   goto 102
	endif
	Xii = Xi
	Xi  = X

	RK = 0.0d0
      enddo
	if(option==2) goto 102
	if(Nq>=10) then
 	  print *, 'cannot converge with the evolution, please:'
	  print *, '1. change the time step to *option 2*,'
	  print *, '2. check your initial conditions, and'
 	  print *, '3. look for shell crossing singularities'
	  print *, '---calculations are being aborted---'
	  stop
	endif	
	if(cti<ctf) then
	Nq = Nq + 1
	goto 101
	endif

 102  continue

	end

!=====================================================

        subroutine get_V(Nx,X,lb,V)
	implicit none
	integer Nx
        double precision, intent(in)  :: X(Nx)
        double precision, intent(out) :: V(Nx)
        double precision lb

! X(1) = density
! X(2) = expansion
! X(3) = shear
! X(4) = Weyl
! NOTE THAT: density and Weyl are x 8pi G/c^2, eg. X(1)=rho*8piG/c^2

	V(1)  = -1.0d0*X(1)*X(2)
	V(2)  = -((X(2)*X(2))/3.0d0)-(X(1)/2.0d0)+lb-6.0*(X(3)*X(3))
	V(3)  = -(2.0d0/3.0d0)*X(2)*X(3)- X(4) + X(3)*X(3)
	V(4)  = -3.0*X(4)*X(3)- X(2)*X(4) - 0.5d0*X(1)*X(3)

        end subroutine

!=====================================================

	subroutine timelcdm(zo,ctt,H_0)
	implicit none
	double precision zo,ct,lb,szpar(30),rhb,rhzo,x,arsh,tzo,ti,ctt
     	double precision ztt,thb
     	double precision, intent(in) :: H_0
	call get_parameters(szpar, H_0)

   	lb = szpar(7) 
	if(lb==0d0) then
	print *, 'Lambda cannot be zero'
	print *, 'if you want to use models with Lambda=0'
	print *, 'then rewrite the subroutine *timelcdm* '
  	print *, '---calculations are being aborted---'
	stop
	endif

         rhzo = szpar(5)*( (1.0d0+zo)**3 )
         x = dsqrt(lb/rhzo)
  	 arsh = dlog(x + dsqrt(x*x + 1d0))
         tzo = (dsqrt((4d0)/(3d0*lb)))*arsh 
         ct = tzo
         ctt = ct

	end

!=====================================================

   	subroutine get_parameters(cpar, H_0)
	implicit none
	double precision cpar(30)
	double precision omega_matter,omega_lambda,H_0,age
	double precision pi, mu,lu,tu,gcons,cs,kap,kapc2,Ho,gkr,lb

	pi = 4d0*datan(1.0d0)

! cosmological parameters / Planck 2015 (TT+lowP+lensing)
	omega_matter = 0.999!0.308
	omega_lambda = 1.0 - omega_matter
	!H0 = 70!67.810d0
	  if(omega_lambda/=(1.0 - omega_matter)) then
	   print *, 'The code uses the LCDM model to set up '
	   print *, ' the initial conditions '
	   print *, 'if you want to use non-LCDM models '
	   print *, 'then change the subroutine *timelcdm* '
  	   print *, '---calculations are being aborted---'
	   stop
	  endif

! units: time in 10^6 years, length in kpc, mass in 10^15 M_{\odot}
	mu=1.989d45
	lu=3.085678d19
	tu=31557600*1d6
! and other constants 
	gcons= 6.6742d-11*((mu*(tu**2))/(lu**3))
	cs=299792458*(tu/lu)
	kap=8d0*pi*gcons*(1d0/(cs**4))
	kapc2=8d0*pi*gcons*(1d0/(cs**2))
	Ho=(tu/(lu))*H_0
	gkr=3d0*(((Ho)**2)/(8d0*pi*gcons))
	lb=3d0*omega_lambda*(((Ho)**2)/(cs*cs))	
	gkr=kapc2*gkr*omega_matter

	cpar(1) = H_0*1d-2
	cpar(2) = Ho/cs
	cpar(3) = omega_matter
	cpar(4) = omega_lambda
	cpar(5) = gkr
	cpar(6) = cs
	cpar(7) = lb
	cpar(8) = pi
	cpar(9) = kapc2

! virialisation type: 1=turnaround, 2=near singularity, 3=stable halo
	cpar(10) = 1.0d0
! for option 2 ("collapsed"), please use either a fixed step (option 2 below), or please decrease the time step -- with default setting the results may not be accurate 

! fix vs dynamical time step: 1=dynamical, 2=fixed
	cpar(11) = 1.d0
! dynamical step does not work well for some extreme cases
! so always test if this choice works well with your system

	end
!=====================================================



