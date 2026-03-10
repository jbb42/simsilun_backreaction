Nice — this is a compact Fortran program that evolves a very small “Silent Universe” model (simsilun). I’ll walk through it top-to-bottom and explain every meaningful line / block so you can see what each piece does and why. I’ll group some closely related lines together where that keeps the explanation clearer (but I’ll still mention every line).

# High-level overview

This program sets up cosmological parameters and simple initial density-contrast data, then evolves each element of that initial vector with a 4-variable ODE system (density, expansion, shear, Weyl) using an explicit RK4 integrator and simple collapse/virialisation handling. Output is the initial density contrast vs final density from the “Silent Universe” evolution and the corresponding linear-EdS prediction.

---

# Program header and declarations

`program simsilun`

* Start of the main program named `simsilun`.

`! code: SIMplified SILent UNiverse, ...` (several `!` comment lines)

* Comments describing the code, author, papers, licence, units — purely informational.

`implicit none`

* Disables implicit typing; every variable must be declared explicitly (safer).

`integer I,Ii`

* Declares two integer scalars `I` and `Ii` (typical loop indices; `Ii` is declared but not used later).

`integer, parameter :: Nx = 4  ! number of variables evolved with the Silent Universe`

* `Nx` is a named constant with value 4: the number of dynamic variables in the ODE system.

`double precision X(Nx)`

* Declares an array `X` of length `Nx` (4) in double precision. The comments below tell us the meaning of each element:

```
! X(1) = density
! X(2) = expansion
! X(3) = shear
! X(4) = Weyl
```

* Additional comment describes the unit scalings used in the code:

```
! density and Weyl are x 8pi G/c^2 
! expansion and shear are x 1/c
```

`integer, parameter :: Ni = 2000  ! dimension of the initial data vector - for single value Ni = 1`

* `Ni` is the number of initial data points to evolve; default here is 2000.

`double precision Din(Ni),dini ! initial density contrast`

* `Din` is an array holding the initial density contrasts (one per sample). `dini` is a scalar used for the current initial density contrast inside the loop.

`double precision Rout(Ni), Reds(Ni) ! final density in Silent Universe and within linearly perturbed Einstein-de Sitter model`

* `Rout` will store the final density from the Silent Universe evolution. `Reds` stores the final density predicted by a linear perturbed Einstein-de Sitter model (for comparison).

`double precision InD(10) ! initial data and the final time instant`

* `InD` is a length-10 double array used to pass various initial and global parameters into subroutines.

`double precision cpar(30)  ! vector with cosmological parameters`

* `cpar` holds cosmological constants and derived parameters (set by `get_parameters`).

---

# Get cosmological parameters and initial data

`! cosmological parameters`
`call get_parameters(cpar)`

* Calls `get_parameters` to fill the `cpar` array with cosmological constants and derived units. `get_parameters` is defined later.

`! load initial data: density contrast vector deli(Ni) and other data in InD(10)`
`call initial_data(cpar,InD,Ni,Din)`

* Calls `initial_data` which sets up `InD` (initial/final times, background densities, options, virialisation mode...) and fills the `Din` array of initial density contrasts. The subroutine uses `cpar`.

---

# Initialize outputs and main loop

`Rout = 0.0d0`
`Reds = 0.0d0`

* Initialize the entire `Rout` and `Reds` arrays to zero.

`! calculate the evolution of X(Nx) -> then -> write density to Xout`
`!$OMP PARALLEL DO PRIVATE(I,dini,X),SHARED(InD,Din,Rout,Reds)`

* An OpenMP parallel-do pragma (comment-style in Fortran) to parallelize the following loop. It lists private/shared variables. If compiled with OpenMP, iterations can run in parallel.

`do I=1,Ni`

* Loop over each initial-data sample index.

`    dini = Din(I)`

* Copy the current initial density contrast into `dini`.

`    call silent_evolution(InD,dini,Nx,X)`

* Evolve the single sample from initial to final time. `silent_evolution` integrates the 4-component ODE for this initial density contrast and returns the final `X` array.

`    Rout(I) = (X(1)/InD(5))`

* Compute and store the final density normalized by final background density: final density (X(1)) divided by `InD(5)` (background density at final time). So `Rout` is the density contrast + 1 at final time from Silent Universe model.

`    Reds(I) = Din(I)*(InD(2)/InD(5))**(1.0/3.0) +1.0`

* Compute the linear/EdS model prediction: initial contrast `Din(I)` scaled by growth factor ratio `(InD(2)/InD(5))^(1/3)` plus 1.0 → stored as `Reds(I)`.

`enddo`
`!$OMP END PARALLEL DO`

* End of loop and OpenMP parallel region.

---

# Write output to file

`! output: initial density contrast ($1) vs final density in the Silent Universe ($2) and Einstein-de Sitter model ($3)`
`open(21,file='density')`

* Open a file unit 21, writing to file named `density`.

`do I=1,Ni`
`   write(21,*) Din(I),Rout(I),Reds(I)`
`enddo`

* Loop over samples and write lines of three values: initial density contrast, final silent-universe density, final EdS-model density. The `*` format is list-directed (default spacing), good for simple numeric columns.

`end`

* End of main program.

---

# Subroutine `initial_data`

`subroutine initial_data(cpar,InD,Ni,Din)`

* Subroutine signature: inputs `cpar`, sets `InD`, `Din`, uses `Ni`.

`implicit none`

* No implicit typing inside the subroutine.

`integer I,Ni`
`double precision InD(10), Din(Ni)`
`double precision cpar(30)`
`double precision zo,zz,zf,cto,ctf`

* Local variable declarations: loop index, arrays and some redshift/time variables.

Comments explain `InD` array indices:

```
! InD(1) = initial time instant
! InD(2) = background's density
! InD(3) = background's expansion rate
! InD(4) = final time instant
! InD(5) = final background's density
! InD(6) = cosmological constant
! InD(7) = virialisation
! InD(8) = time step
```

`! initial values: redshift, time instant, density, and expansion rate (the LCDM model assumed)`
`zo = 1090.0d0`

* `zo` = 1090 — the initial redshift (roughly last-scattering, CMB). They start evolution from z=1090.

`zz = (zo+1.0d0)`

* `zz = 1 + z` used in matter density scaling.

`call timelcdm(zo,cto)`

* Call `timelcdm` to get cosmic time `cto` corresponding to initial redshift `zo` in the LCDM model.

`InD(1) = cto`

* Store initial cosmic time in `InD(1)`.

`InD(2) = cpar(5)*(zz**3)`

* Background density at initial time: `cpar(5)` appears to be matter density normalization; multiplied by `(1+z)^3`.

`InD(3) = 3.0d0*cpar(2)*dsqrt(cpar(3)*(zz**3) + cpar(4))`

* Background expansion rate at initial time. Here `cpar(2)` is H/c in code units, `cpar(3)` is Ω\_m, `cpar(4)` is Ω\_Λ. The expression is `3 * H * sqrt(Ω_m (1+z)^3 + Ω_Λ)` in the code's units.

`! final time instants`
`zf = 0.0`

* Final redshift `zf = 0` (today).

`call timelcdm(zf,ctf)`
`InD(4) = ctf`

* Get cosmic time at final redshift (today) and store in `InD(4)`.

`zz = (zf+1.0d0)`
`InD(5) = cpar(5)*(zz**3)`

* Final background density (z=0): `cpar(5)` × (1)^3 → essentially `cpar(5)`.

`! initial vector with density contrasts`
`!Generate a simple example of initial conditions.`
`do I=1,Ni`
`Din(I) = -0.00095+0.000001*I`
`enddo`

* This fills `Din` with a toy linear sequence of initial density contrasts starting from `-9.5e-4` and incrementing by `1e-6` each step. The comment indicates this is placeholder data; real runs would read from a file (e.g., Millennium initial conditions).

`! other parameters`
`InD(6) = cpar(7)`
`InD(7) = cpar(10)`
`InD(8) = cpar(11)`

* Copies some parameters from cosmology array into `InD`: `InD(6)` is cosmological constant `lb`; `InD(7)` is virialisation mode; `InD(8)` is the time-stepping option.

`end`

* End of `initial_data` subroutine.

---

# Subroutine `silent_evolution` — the ODE integrator

`subroutine silent_evolution(InD,dini,Nx,X)`

* Integrates the 4-variable silent-universe ODE from initial time to final time for a single initial density contrast `dini`.

`implicit none`

* No implicit typing.

`integer I,J, Nx,Nf,Nq`
`integer option, virialisation`
`double precision, intent(in) :: InD(10), dini`
`double precision X(Nx),Xi(Nx),Xii(Nx),V(Nx),RK(Nx,4)`
`double precision xp,xp1,xp2,xp3,yp(Nx),yp1(Nx),yp2(Nx),yp3(Nx)`
`double precision lb,tevo,dt,cti,cto,ctf`
`logical collapse`

* Many local vars: `Xi` and `Xii` hold previous values for interpolation/rollback; `RK` stores Runge–Kutta stage increments; `yp*` arrays help with Lagrange interpolation when overshoot happens; `lp` time markers for interpolation; `collapse` flag.

`! parameters`
`lb = InD(6)`

* Local copy of cosmological constant (perhaps used in ODEs).

`virialisation = int(InD(7))`

* Virialisation mode (integer): how to handle collapse.

`option = int(Ind(8))`

* Time-step option (1 = dynamical adaptive step, 2 = fixed steps). **Note:** small typo in code: `Ind` instead of `InD` — Fortran is case-insensitive, but spelling must match: if in source it really is `Ind` while argument is `InD`, Fortran will treat them same if same letters; but if it's actually mismatched in whitespace, compilation might fail. In practice this works because Fortran ignores case; but if source had a real typo it's risky. (I'll assume it's intended `InD`.)

`collapse = .false.`

* Initialize collapse flag to false.

`! time of integration, and other time instants:`
`cto = InD(1)`
`ctf = InD(4)`
`tevo = ctf - cto`
`cti = cto`

* `cto`=start time, `ctf`=final time, `tevo`=total evolution time, `cti`=current time initialized to start.

`xp1 = cto`
`xp2 = cto`
`xp3 = cto`

* Initialize three previous time samples used for Lagrange interpolation if final time is overshot.

`! initial conditions`
`X(1) = InD(2)*(1.0d0 + dini        )`
`X(2) = InD(3)*(1.0d0 -(dini/3.0d0) )`
`X(3) =  (dini/9.0d0)*InD(3)`
`X(4) = -(dini/6.0d0)*InD(2)`

* Set initial values for the 4 dynamic variables from background plus small perturbations linearized in `dini`. These are perturbation relations from the Silent Universe approximation (density, expansion, shear, Weyl).

`call get_V(Nx,X,lb,V)`

* Compute derivatives `V` = dX/dt at the initial state using `get_V` subroutine (the ODE right-hand side).

`Xi = X`
`Xii= X`

* Copy the initial `X` to previous-state arrays `Xi` and `Xii`.

`! integration steps (see also get_parameters for options 1 and 2)`
`if(option.eq.1) then`
`  dt = dabs(1d-3/(X(2) + 0.33*X(3)))`
`  Nf = 1000*int(1000.0*tevo*(X(2) + 0.33*X(3)))`
`endif`

* If option 1 (dynamical step): choose `dt` based on current expansion+shear to keep step adaptive. `Nf` is estimated number of steps from total evolution time and local expansion rate; the factor `1000*int(1000.0*...)` is an odd way to get a big integer scaling—basically sets `Nf` proportional to `tevo*(X(2)+0.33*X(3))`.

`if(option.eq.2) then`
`  Nf = 350000`
`  dt = tevo/(1.0d0*Nf)`
`endif`

* If option 2 (fixed-step): use a fixed number `Nf=350000` steps and set `dt` accordingly.

`if(option.ne.1 .and. option.ne.2) then`
`  print *, 'please specify the *option* for the time step'`
`  print *, '---calculations are being aborted---'`
`  stop`
`endif`

* If option is not 1 or 2, abort with a message.

`Nq = 0`

* `Nq` is a retry counter used later if the integrator did not get to final time (loop repeats).

`101  continue`

* A labelled point used by the algorithm to retry the integration if cti < ctf at end.

---

## The RK4 integration loop

`do I=1,Nf`

* Loop over `Nf` RK steps.

`  if(option.eq.1) dt = dabs(1d-3/(X(2) + X(3)/3.0d0))`
`  if(option.eq.2) dt = tevo/(1.0d0*Nf)`

* Update `dt` each iteration depending on the chosen option. For option 1, a slightly different expression for dt than earlier (`X(3)/3` vs `0.33*X(3)`), still adaptive.

`  cti = cti + dt`

* Advance current time by `dt`.

`  call get_V(Nx,X,lb,V)`
`  do J=1,Nx`
`    RK(J,1) = dt*V(J)`
`    X(J) = Xi(J) + 0.5*RK(J,1)`
`  enddo`

* First RK stage: compute derivative `V` at current `X`, compute stage increment `RK(:,1)` and set temporary `X` to midpoint estimate.

`  call get_V(Nx,X,lb,V)`
`  do J=1,Nx`
`    RK(J,2) = dt*V(J)`
`    X(J) = Xi(J) + 0.5*RK(J,2)`
`  enddo`

* Second RK stage: compute derivative at the midpoint, set `X` again for next stage.

`  call get_V(Nx,X,lb,V)`
`  do J=1,Nx`
`    RK(J,3) = dt*V(J)`
`    X(J) = Xi(J) + RK(J,3)`
`  enddo`

* Third RK stage: compute derivative at second midpoint and set `X` to the full-step estimate.

`  call get_V(Nx,X,lb,V)`
`  do J=1,Nx`
`    RK(J,4) = dt*V(J)`
`    X(J)=Xi(J)+(RK(J,1)+2.0*(RK(J,2)+RK(J,3))+RK(J,4))/6.0d0`
`  enddo`

* Fourth stage and combine RK increments to update `X` from `Xi` to the new `X` using the standard RK4 formula.

---

## Collapse detection and virialisation handling

`! check for the collapse and apply virialisation if necessary`
`if(X(2).le.0.0d0) collapse = .true.`

* If expansion `X(2)` becomes ≤ 0, flag collapse (region stops expanding and collapses).

`if(collapse) then`

* Several possible virialisation behaviours depending on `virialisation` value:

`  if(virialisation.eq.1) then`
`    X(2) = 0.0d0`
`    goto 102`
`  endif`

* If virialisation type 1: set expansion to zero (turnaround), then jump to label `102` to exit integration.

`  if(virialisation.eq.2) then`
`    if(isnan(X(1)) .or. isnan(X(2))) then`
`      X = Xii`
`      X(2) = 0.0d0`
`      goto 102`
`    endif`
`  endif`

* Virialisation type 2: if any NaNs appear in density or expansion, revert to previous saved `Xii` (rollback) and set expansion to zero, then exit. This is a safety mechanism to handle numerical blow-ups.

`  if(virialisation.eq.3) then`
`    X(1) = (InD(5)*60.0d0)`
`    X(2) = 0.0d0`
`    X(3) = 0.0d0`
`    X(4) = 0.0d0`
`    goto 102`
`  endif`

* Virialisation type 3: force the system into a “stable halo” with density = 60 × background (arbitrary choice here), and zero expansion, shear, Weyl — then exit.

`endif`

* End collapse handling.

---

## Final-time overshoot correction with Lagrange interpolation

A comment explains the integrator may overshoot final time; they perform Lagrange interpolation to get values exactly at `ctf`.

`xp1 = xp2`
`xp2 = xp3`
`xp3 = cti`
`xp  = ctf`

* Shift time-history variables: `xp1`, `xp2`, `xp3` hold the previous three time values; `xp` is target final time.

`if(cti.eq.ctf) goto 102`

* If we landed exactly at final time, jump to exit label `102`.

`if(cti.gt.ctf) then`
`  yp1 = Xii`
`  yp2 = Xi`
`  yp3 = X`
`  X = 0.0d0`
`  X = X + yp1*((xp - xp2)/(xp1-xp2))*((xp - xp3)/(xp1-xp3))`
`  X = X + yp2*((xp - xp1)/(xp2-xp1))*((xp - xp3)/(xp2-xp3))`
`  X = X + yp3*((xp - xp1)/(xp3-xp1))*((xp - xp2)/(xp3-xp2))`
`  goto 102`
`endif`

* If we overshot final time, use 3-point Lagrange interpolation on the three stored states `yp1, yp2, yp3` (values at times `xp1,xp2,xp3`) to evaluate `X` exactly at `xp` = `ctf`. Then exit.

`Xii = Xi`
`Xi  = X`

* If not yet at final time, move the step history forward: `Xii` ≤ `Xi` ≤ `X`.

`RK = 0.0d0`

* Clear the RK array for next step.

`enddo`

* End RK loop.

---

## Post-integration convergence check and retries

`if(option.eq.2) goto 102`

* If fixed-step option 2 was used, don't try to refine: jump to exit label.

`if(Nq.ge.10) then`
`  print *, 'cannot converge with the evolution, please:'`
`  print *, '1. change the time step to *option 2*,'`
`  print *, '2. check your initial conditions, and'`
`  print *, '3. look for shell crossing singularities'`
`  print *, '---calculations are being aborted---'`
`  stop`
`endif`

* If we've retried more than 10 times without integrating to final time, abort and print suggestions.

`if(cti.lt.ctf) then`
`  Nq = Nq + 1`
`  goto 101`
`endif`

* If we didn't reach final time (`cti` still less than `ctf`), increment retry counter and jump back to label `101` to re-run the integration (this allows adjusting `Nf`/`dt` if option 1 is used, though exact behavior depends on earlier computations).

`102  continue`

* Exit label for successful end-of-integration and after interpolation/virialisation corrections.

`end`

* End subroutine `silent_evolution`.

---

# Subroutine `get_V` — the right-hand side (ODEs)

`subroutine get_V(Nx,X,lb,V)`

* Computes the time derivatives `V` = dX/dt for the 4 variables.

`implicit none`
`integer Nx`
`double precision, intent(in)  :: X(Nx)`
`double precision, intent(out) :: V(Nx)`
`double precision lb`

* Declarations. `lb` is cosmological constant.

Comments remind variable meanings and units.

`V(1)  = -1.0d0*X(1)*X(2)`

* `d(rho)/dt = -rho * expansion` (mass conservation; consistent with fluid equation in comoving time).

`V(2)  = -((X(2)*X(2))/3.0d0)-(X(1)/2.0d0)+lb-6.0*(X(3)*X(3))`

* Evolution equation for expansion (Raychaudhuri-like): negative quadratic term from expansion, gravitational focusing from density `-X(1)/2`, cosmological-constant term `+lb`, and shear term `-6 σ^2`.

`V(3)  = -(2.0d0/3.0d0)*X(2)*X(3)- X(4) + X(3)*X(3)`

* Shear evolution: coupling to expansion (`-2/3 expansion * shear`), Weyl curvature `-X(4)`, and shear self-interaction `+shear^2`.

`V(4)  = -3.0*X(4)*X(3)- X(2)*X(4) - 0.5d0*X(1)*X(3)`

* Weyl tensor (tidal field) evolution: coupling to shear and expansion and a source term from density × shear.

`end subroutine`

* End RHS subroutine.

---

# Subroutine `timelcdm` — cosmic time for LCDM

`subroutine timelcdm(zo,ctt)`

* Computes cosmic time `ctt` (in the code's time units) corresponding to redshift `zo`, in the assumed LCDM model.

`implicit none`
`double precision zo,ct,lb,szpar(30),rhb,rhzo,x,arsh,tzo,ti,ctt`
`double precision ztt,thb`
`call get_parameters(szpar)`

* Local variables and call to `get_parameters` to get cosmological constants.

`lb = szpar(7)`
`if(lb.eq.0d0) then`
`  print *, 'Lambda cannot be zero'`
`  print *, 'if you want to use models with Lambda=0'`
`  print *, 'then rewrite the subroutine *timelcdm* '`
`  print *, '---calculations are being aborted---'`
`  stop`
`endif`

* Requires non-zero Λ; code is written for ΛCDM only.

`rhzo = szpar(5)*( (1.0d0+zo)**3 )`
`x = dsqrt(lb/rhzo)`
`arsh = dlog(x + dsqrt(x*x + 1d0))`
`tzo = (dsqrt((4d0)/(3d0*lb)))*arsh`
`ct = tzo`
`ctt = ct`

* Computes an analytic expression for the cosmic time at redshift `zo` in closed form using inverse hyperbolic sine/arcsinh formula (or arsh via log). The expression derives from the Λ-dominated Friedmann solution for time as a function of scale factor. The time units are the code's internal unit system (see `get_parameters`).

`end`

* End `timelcdm`.

---

# Subroutine `get_parameters` — units and cosmology

`subroutine get_parameters(cpar)`

* Fills `cpar` with cosmology and unit conversion constants.

`implicit none`
`double precision cpar(30)`
`double precision omega_matter,omega_lambda,H0,age`
`double precision pi, mu,lu,tu,gcons,cs,kap,kapc2,Ho,gkr,lb`

* Local variables.

`pi = 4d0*datan(1.0d0)`

* Compute π.

`! cosmological parameters / Planck 2015 (TT+lowP+lensing)`
`omega_matter = 0.308`
`omega_lambda = 1.0 - omega_matter`
`H0 = 67.810d0`

* Set Ω\_m, Ω\_Λ (complement), and H0 (in km/s/Mpc). They mention Planck 2015 numbers.

`if(omega_lambda.ne.(1.0 - omega_matter)) then`
`  print *, 'The code uses the LCDM model to set up '`
`  print *, ' the initial conditions '`
`  print *, 'if you want to use non-LCDM models '`
`  print *, 'then change the subroutine *timelcdm* '`
`  print *, '---calculations are being aborted---'`
`  stop`
`endif`

* Sanity check: they expect Ω\_Λ to be exactly `1 - Ω_m`.

`! units: time in 10^6 years, length in kpc, mass in 10^15 M_{\odot}`
`mu=1.989d45`
`lu=3.085678d19`
`tu=31557600*1d6`

* Unit constants: `mu` mass unit in code (10^15 M☉ in kg?), `lu` length unit (kpc in meters), `tu` time unit (10^6 years in seconds).

`! and other constants`
`gcons= 6.6742d-11*((mu*(tu**2))/(lu**3))`

* Newton constant `G` in code units, scaled by the unit choices.

`cs=299792458*(tu/lu)`

* Speed of light in code units (c × time/length unit).

`kap=8d0*pi*gcons*(1d0/(cs**4))`
`kapc2=8d0*pi*gcons*(1d0/(cs**2))`

* κ = 8πG/c^4 and κ c^2 = 8πG/c^2 used for unit scaling.

`Ho=(tu/(lu))*H0`

* Hubble constant `H0` transformed from km/s/Mpc to code time/length units.

`gkr=3d0*(((Ho)**2)/(8d0*pi*gcons))`
`lb=3d0*omega_lambda*(((Ho)**2)/(cs*cs))`
`gkr=kapc2*gkr*omega_matter`

* Compute background critical density, cosmological constant `lb` in code units, and `gkr` is some derived density (probably background matter density in code units).

Now assign `cpar` entries:

```
cpar(1) = H0*1d-2
cpar(2) = Ho/cs
cpar(3) = omega_matter
cpar(4) = omega_lambda
cpar(5) = gkr
cpar(6) = cs
cpar(7) = lb
cpar(8) = pi	
cpar(9) = kapc2
```

* `cpar(1)` small H0 scaling (not used earlier). `cpar(2)` = Ho/cs is H/c in code units. `cpar(5)` stores the background density normalization used by `initial_data`. `cpar(7)` is cosmological-constant `lb`. `cpar(10)` and `cpar(11)` are set below.

`! virialisation type: 1=turnaround, 2=near singularity, 3=stable halo`
`cpar(10) = 1.0d0`

* Default virialisation is type 1 (turnaround).

`! fix vs dynamical time step: 1=dynamical, 2=fixed`
`cpar(11) = 1.d0`

* Default time-step option is 1 (dynamical/adaptive).

`end`

* End `get_parameters`.

---

# A few practical notes and caveats

* Units: the code uses bespoke units (time in 10^6 years, length in kpc, mass in 10^15 M⊙). `get_parameters` converts physical constants to those units; be careful when interpreting numeric outputs.
* Numerical methods: explicit RK4 with adaptive or fixed stepping. Option 1 (adaptive) uses heuristics based on expansion+shear; option 2 is fixed-step and may be more stable for some systems.
* Collapse/virialisation: the code attempts simple remedies when collapse occurs: set expansion to zero (turnaround), rollback on NaNs, or force a stable high-density halo. These are approximations — a more physical treatment would require full hydrodynamics or an N-body/halo model.
* Lagrange interpolation: used to avoid overshooting the exact final time `ctf`. The algorithm keeps three previous states and time stamps to interpolate to the exact requested final time.
* Parallelism: the outer loop over `Ni` samples is OpenMP-parallelizable — each sample is independent so this is a sensible parallelization. Make sure to compile with OpenMP and that `get_parameters`/`initial_data` are thread-safe (they are, because only reading `cpar`).

---

If you like, I can:

* annotate the code in-place with commented explanations (so you can keep it as a documented source file),
* point out small bugs/typos to fix (I noticed `Ind(8)` vs `InD(8)` and some odd dt/Nf scaling),
* translate the integrator to Python/Julia for experimentation, or
* extract the ODE system and show phase-space plots for a sample initial condition.

Which of those would you like next?
