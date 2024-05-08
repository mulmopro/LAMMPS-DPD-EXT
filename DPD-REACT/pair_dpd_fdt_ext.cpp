// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Nunzia Lauriello (Politecnico di Torino)
   Contributing author: James Larentzos (U.S. Army Research Laboratory)   
------------------------------------------------------------------------- */

#include "pair_dpd_fdt_ext.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "fix.h"
#include "force.h"
#include "memory.h"
#include "modify.h"
#include "neigh_list.h"
#include "neighbor.h"
#include "random_mars.h"
#include "update.h"

#include <cmath>

using namespace LAMMPS_NS;

#define EPSILON 1.0e-10

/* ---------------------------------------------------------------------- */

PairDPDfdtExt::PairDPDfdtExt(LAMMPS *lmp) : Pair(lmp)
{
  random = nullptr;
  splitFDT_flag = false;
  a0_is_zero = false;
  eflag_either = 1;
  vflag_global = vflag_atom = cvflag_atom =1;
}

/* ---------------------------------------------------------------------- */

PairDPDfdtExt::~PairDPDfdtExt()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(cutD);
    memory->destroy(a0);
    memory->destroy(sigma);
    memory->destroy(sigmaT);
    memory->destroy(ws);
    memory->destroy(wsT);
    memory->destroy(cut);
  }


  if (random) delete random;
}

/* ---------------------------------------------------------------------- */

void PairDPDfdtExt::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpairx,fpairy,fpairz,fpair,fpairDx,fpairDy,fpairDz,fpairD,fpairRx,fpairRy,fpairRz,fpairR;
  double vxtmp,vytmp,vztmp,delvx,delvy,delvz;
  double rsq,r,rinv,dot,wd,wdc,wdPar,wdPerp,randnum,randnumx,randnumy,randnumz,factor_dpd;
  int *ilist,*jlist,*numneigh,**firstneigh;
  double P[3][3];
  double gamma_ij, gammaT_ij;

  evdwl = 0.0;
  ev_init(eflag,vflag);

  double **x = atom->x;
  double **v = atom->v;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;
  double dtinvsqrt = 1.0/sqrt(update->dt);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  if (splitFDT_flag) {
    if (!a0_is_zero) for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      vxtmp = v[i][0];
      vytmp = v[i][1];
      vztmp = v[i][2];
      itype = type[i];
      jlist = firstneigh[i];
      jnum = numneigh[i];

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        factor_dpd = special_lj[sbmask(j)];
        j &= NEIGHMASK;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        jtype = type[j];

        // if (rsq < cutsq[itype][jtype]) {
	if (rsq < cutD[itype][jtype]*cutD[itype][jtype]) {
          r = sqrt(rsq);
       	  if (r < EPSILON) continue;  // r can be 0.0 in DPD systems
       	  rinv = 1.0/r;
      	  delvx = vxtmp - v[j][0];
          delvy = vytmp - v[j][1];
          delvz = vztmp - v[j][2];
          dot = delx*delvx + dely*delvy + delz*delvz; 
          
      wd=1.0-r/cutD[itype][jtype];
	  wdPar = pow(wd,ws[itype][jtype]);
	  wdPerp = pow(wd,wsT[itype][jtype]);
          
               
          
	  if (r <=cut[itype][jtype]) wdc = 1.0 -r/cut[itype][jtype];
	  else wdc=0.0;
	  
	  P[0][0] = 1.0 - delx*delx*rinv*rinv;
	  P[0][1] =     - delx*dely*rinv*rinv;
	  P[0][2] =     - delx*delz*rinv*rinv;

	  P[1][0] = P[0][1];
	  P[1][1] = 1.0 - dely*dely*rinv*rinv;
	  P[1][2] =     - dely*delz*rinv*rinv;

	  P[2][0] = P[0][2];
	  P[2][1] = P[1][2];
	  P[2][2] = 1.0 - delz*delz*rinv*rinv;

          randnum = random->gaussian();
	  randnumx = random->gaussian();
	  randnumy = random->gaussian();
	  randnumz = random->gaussian();
          gamma_ij = sigma[itype][jtype]*sigma[itype][jtype]
                     / (2.0*force->boltz*temperature);
	  gammaT_ij = sigmaT[itype][jtype]*sigmaT[itype][jtype]
                     / (2.0*force->boltz*temperature);  
	  
          // conservative force = a0 * wdc
          fpair = a0[itype][jtype]*wdc;
          
          
          fpairx = fpair*rinv*delx;
	  	  fpairy = fpair*rinv*dely;
	      fpairz = fpair*rinv*delz;
	      
	      fpairx *= factor_dpd;
	      fpairy *= factor_dpd;
	      fpairz *= factor_dpd;
	      

          f[i][0] += fpairx;
          f[i][1] += fpairy;
          f[i][2] += fpairz;
          if (newton_pair || j < nlocal) {
            f[j][0] -= fpairx;
            f[j][1] -= fpairy;
            f[j][2] -= fpairz;
          }
          
       
       //DRAG FORCE AND RANDOM FORCE EVALUATION
       
       
       // drag force - parallel
       
      fpairD = -gamma_ij*wdPar*wdPar*dot*rinv;

	  // random force - parallel
	  fpairR =  sigma[itype][jtype]*wdPar*randnum*dtinvsqrt;

	  fpairDx = fpairD*rinv*delx;
	  fpairDy = fpairD*rinv*dely;
	  fpairDz = fpairD*rinv*delz;
	  
	  fpairRx = fpairR*rinv*delx;
	  fpairRy = fpairR*rinv*dely;
	  fpairRz = fpairR*rinv*delz;

	  // random force - parallel
	  /*fpairx += sigmaT[itype][jtype]*wdPar*
                    ((-P[0][0]+1)*randnumx - P[0][1]*randnumy - P[0][2]*randnumz)*dtinvsqrt;
            fpairy += sigmaT[itype][jtype]*wdPar*
                    (-P[1][0]*randnumx + (-P[1][1]+1)*randnumy - P[1][2]*randnumz)*dtinvsqrt;
            fpairz += sigmaT[itype][jtype]*wdPar*
                    (-P[2][0]*randnumx - P[2][1]*randnumy + (-P[2][2]+1)*randnumz)*dtinvsqrt;*/

          // drag force - perpendicular
	  fpairDx -= gammaT_ij*wdPerp*wdPerp*
                    (P[0][0]*delvx + P[0][1]*delvy + P[0][2]*delvz);
	  fpairDy -= gammaT_ij*wdPerp*wdPerp*
                    (P[1][0]*delvx + P[1][1]*delvy + P[1][2]*delvz);
	  fpairDz -= gammaT_ij*wdPerp*wdPerp*
                    (P[2][0]*delvx + P[2][1]*delvy + P[2][2]*delvz);

          // random force - perpendicular
	  fpairRx += sigmaT[itype][jtype]*wdPerp*
                    (P[0][0]*randnumx + P[0][1]*randnumy + P[0][2]*randnumz)*dtinvsqrt;
	  fpairRy += sigmaT[itype][jtype]*wdPerp*
                    (P[1][0]*randnumx + P[1][1]*randnumy + P[1][2]*randnumz)*dtinvsqrt;
	  fpairRz += sigmaT[itype][jtype]*wdPerp*
                    (P[2][0]*randnumx + P[2][1]*randnumy + P[2][2]*randnumz)*dtinvsqrt;
        
	  fpairDx *= factor_dpd;
	  fpairDy *= factor_dpd;
	  fpairDz *= factor_dpd;
      fpairRx *= factor_dpd;
	  fpairRy *= factor_dpd;
	  fpairRz *= factor_dpd; 
      
          if (eflag) {
            // unshifted eng of conservative term:
            // evdwl = -a0[itype][jtype]*r * (1.0-0.5*r/cut[itype][jtype]);
            // eng shifted to 0.0 at cutoff
            evdwl = 0.5*a0[itype][jtype]*cut[itype][jtype]*wdc * wdc;
            evdwl *= factor_dpd;
          }

          if (evflag) ev_tally_xyz_SH (i,j,nlocal,newton_pair,evdwl,0.0,fpairx,fpairy,fpairz, fpairDx,fpairDy,fpairDz, fpairRx,fpairRy,fpairRz, delx,dely,delz);
        }
      }
    }
  } else {
    for (ii = 0; ii < inum; ii++) {
      i = ilist[ii];
      xtmp = x[i][0];
      ytmp = x[i][1];
      ztmp = x[i][2];
      vxtmp = v[i][0];
      vytmp = v[i][1];
      vztmp = v[i][2];
      itype = type[i];
      jlist = firstneigh[i];
      jnum = numneigh[i];

      for (jj = 0; jj < jnum; jj++) {
        j = jlist[jj];
        factor_dpd = special_lj[sbmask(j)];
        j &= NEIGHMASK;

        delx = xtmp - x[j][0];
        dely = ytmp - x[j][1];
        delz = ztmp - x[j][2];
        rsq = delx*delx + dely*dely + delz*delz;
        jtype = type[j];

        // if (rsq < cutsq[itype][jtype]) {
	if (rsq < cutD[itype][jtype]*cutD[itype][jtype]) {
          r = sqrt(rsq);
          if (r < EPSILON) continue;     // r can be 0.0 in DPD systems
          rinv = 1.0/r;
          delvx = vxtmp - v[j][0];
          delvy = vytmp - v[j][1];
          delvz = vztmp - v[j][2];
          dot = delx*delvx + dely*delvy + delz*delvz;
          // wr = 1.0 - r/cut[itype][jtype];
          // wd = wr*wr;
	  wd=1.0-r/cutD[itype][jtype];
	  wdPar = pow(wd,ws[itype][jtype]);
	  wdPerp = pow(wd,wsT[itype][jtype]);
	  if (r <=cut[itype][jtype]) wdc = 1.0 -r/cut[itype][jtype];
	  else wdc=0.0;

	  P[0][0] = 1.0 - delx*delx*rinv*rinv;
	  P[0][1] =     - delx*dely*rinv*rinv;
	  P[0][2] =     - delx*delz*rinv*rinv;

	  P[1][0] = P[0][1];
	  P[1][1] = 1.0 - dely*dely*rinv*rinv;
	  P[1][2] =     - dely*delz*rinv*rinv;

	  P[2][0] = P[0][2];
	  P[2][1] = P[1][2];
	  P[2][2] = 1.0 - delz*delz*rinv*rinv;

          randnum = random->gaussian();
	  randnumx = random->gaussian();
	  randnumy = random->gaussian();
	  randnumz = random->gaussian();
          gamma_ij = sigma[itype][jtype]*sigma[itype][jtype]
                     / (2.0*force->boltz*temperature);
	  gammaT_ij = sigmaT[itype][jtype]*sigmaT[itype][jtype]
                     / (2.0*force->boltz*temperature);

          // conservative force = a0 * wdc
          // drag force = -gamma * wd^2 * (delx dot delv) / r
          // random force = sigma * wd * rnd * dtinvsqrt;

          fpair = a0[itype][jtype]*wdc;
	  // drag force - parallel
          fpair -= gamma_ij*wdPar*wdPar*dot*rinv;

	  // random force - parallel
	  fpair += sigma[itype][jtype]*wdPar*randnum*dtinvsqrt;

	  fpairx = fpair*rinv*delx;
	  fpairy = fpair*rinv*dely;
	  fpairz = fpair*rinv*delz;

	  // random force - parallel
	  /*fpairx += sigmaT[itype][jtype]*wdPar*
                    ((-P[0][0]+1)*randnumx - P[0][1]*randnumy - P[0][2]*randnumz)*dtinvsqrt;
            fpairy += sigmaT[itype][jtype]*wdPar*
                    (-P[1][0]*randnumx + (-P[1][1]+1)*randnumy - P[1][2]*randnumz)*dtinvsqrt;
            fpairz += sigmaT[itype][jtype]*wdPar*
                    (-P[2][0]*randnumx - P[2][1]*randnumy + (-P[2][2]+1)*randnumz)*dtinvsqrt;*/

          // drag force - perpendicular
	  fpairx -= gammaT_ij*wdPerp*wdPerp*
                    (P[0][0]*delvx + P[0][1]*delvy + P[0][2]*delvz);
	  fpairy -= gammaT_ij*wdPerp*wdPerp*
                    (P[1][0]*delvx + P[1][1]*delvy + P[1][2]*delvz);
	  fpairz -= gammaT_ij*wdPerp*wdPerp*
                    (P[2][0]*delvx + P[2][1]*delvy + P[2][2]*delvz);

          // random force - perpendicular
	  fpairx += sigmaT[itype][jtype]*wdPerp*
                    (P[0][0]*randnumx + P[0][1]*randnumy + P[0][2]*randnumz)*dtinvsqrt;
	  fpairy += sigmaT[itype][jtype]*wdPerp*
                    (P[1][0]*randnumx + P[1][1]*randnumy + P[1][2]*randnumz)*dtinvsqrt;
	  fpairz += sigmaT[itype][jtype]*wdPerp*
                    (P[2][0]*randnumx + P[2][1]*randnumy + P[2][2]*randnumz)*dtinvsqrt;
        
	  fpairx *= factor_dpd;
	  fpairy *= factor_dpd;
	  fpairz *= factor_dpd;

          f[i][0] += fpairx;
          f[i][1] += fpairy;
          f[i][2] += fpairz;
          if (newton_pair || j < nlocal) {
	    f[j][0] -= fpairx;
	    f[j][1] -= fpairy;
	    f[j][2] -= fpairz;
          }

          if (eflag) {
            // unshifted eng of conservative term:
            // evdwl = -a0[itype][jtype]*r * (1.0-0.5*r/cut[itype][jtype]);
            // eng shifted to 0.0 at cutoff
            evdwl = 0.5*a0[itype][jtype]*cut[itype][jtype] * wdc*wdc;
            evdwl *= factor_dpd;
          }

          if (evflag) ev_tally_xyz(i,j,nlocal,newton_pair,evdwl,0.0,fpairx,fpairy,fpairz, delx,dely,delz);
        }
      }
    }
  }
  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairDPDfdtExt::allocate()
{
  allocated = 1;
  int n = atom->ntypes;
  
  

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(cutD,n+1,n+1,"pair:cutD");
  memory->create(a0,n+1,n+1,"pair:a0");
  memory->create(sigma,n+1,n+1,"pair:sigma");
  memory->create(sigmaT,n+1,n+1,"pair:sigmaT");
  memory->create(ws,n+1,n+1,"pair:ws");
  memory->create(wsT,n+1,n+1,"pair:wsT");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairDPDfdtExt::settings(int narg, char **arg)
{
  // process keywords
  if (narg != 3) error->all(FLERR,"Illegal pair_style command");

  temperature = utils::numeric(FLERR,arg[0],false,lmp);
  cut_global = utils::numeric(FLERR,arg[1],false,lmp);
  seed = utils::inumeric(FLERR,arg[2],false,lmp);

  // initialize Marsaglia RNG with processor-unique seed

  if (seed <= 0) error->all(FLERR,"Illegal pair_style command");
  delete random;
  random = new RanMars(lmp,seed + comm->me);

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i; j <= atom->ntypes; j++)
        if (setflag[i][j]) cutD[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairDPDfdtExt::coeff(int narg, char **arg)
{
  if (narg < 8 || narg > 9) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error);
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error);

  double a0_one = utils::numeric(FLERR,arg[2],false,lmp);
  double sigma_one = utils::numeric(FLERR,arg[3],false,lmp);
  double sigmaT_one = utils::numeric(FLERR,arg[4],false,lmp);
  double ws_one = utils::numeric(FLERR,arg[5],false,lmp);
  double wsT_one = utils::numeric(FLERR,arg[6],false,lmp);
  double cut_one = utils::numeric(FLERR,arg[7],false,lmp);
  double cutD_one = cut_global;

  a0_is_zero = (a0_one == 0.0); // Typical use with SSA is to set a0 to zero

  if (narg == 9) cutD_one = utils::numeric(FLERR,arg[8],false,lmp);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      a0[i][j] = a0_one;
      sigma[i][j] = sigma_one;
      sigmaT[i][j] = sigmaT_one;
      ws[i][j] = ws_one;
      wsT[i][j] = wsT_one;
      cut[i][j] = cut_one;
      cutD[i][j] = cutD_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairDPDfdtExt::init_style()
{
  if (comm->ghost_velocity == 0)
    error->all(FLERR,"Pair dpdext/fdt requires ghost atoms store velocity");

  splitFDT_flag = false;
  neighbor->add_request(this);
  for (int i = 0; i < modify->nfix; i++)
    if (utils::strmatch(modify->fix[i]->style,"^shardlow")) {
      splitFDT_flag = true;
    }

  // if newton off, forces between atoms ij will be double computed
  // using different random numbers if splitFDT_flag is false
  if (!splitFDT_flag && (force->newton_pair == 0) && (comm->me == 0)) error->warning(FLERR,
      "Pair dpdext/fdt requires newton pair on if not also using fix shardlow");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairDPDfdtExt::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");

  cut[j][i] = cut[i][j];
  cutD[j][i] = cutD[i][j];
  a0[j][i] = a0[i][j];
  sigma[j][i] = sigma[i][j];
  sigmaT[j][i] = sigmaT[i][j];
  ws[j][i] = ws[i][j];
  wsT[j][i] = wsT[i][j];
  
  return cutD[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairDPDfdtExt::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&a0[i][j],sizeof(double),1,fp);
        fwrite(&sigma[i][j],sizeof(double),1,fp);
        fwrite(&sigmaT[i][j],sizeof(double),1,fp);
        fwrite(&ws[i][j],sizeof(double),1,fp);
        fwrite(&wsT[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
	fwrite(&cutD[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairDPDfdtExt::read_restart(FILE *fp)
{
  read_restart_settings(fp);

  allocate();

  a0_is_zero = true; // start with assumption that a0 is zero
  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) utils::sfread(FLERR,&setflag[i][j],sizeof(int),1,fp,nullptr,error);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          utils::sfread(FLERR,&a0[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&sigma[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&cut[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&cutD[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&sigmaT[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&ws[i][j],sizeof(double),1,fp,nullptr,error);
          utils::sfread(FLERR,&wsT[i][j],sizeof(double),1,fp,nullptr,error);
        }
        MPI_Bcast(&a0[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigma[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cutD[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigmaT[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&ws[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&wsT[i][j],1,MPI_DOUBLE,0,world);
        a0_is_zero = a0_is_zero && (a0[i][j] == 0.0); // verify the zero assumption
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairDPDfdtExt::write_restart_settings(FILE *fp)
{
  fwrite(&temperature,sizeof(double),1,fp);
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&seed,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairDPDfdtExt::read_restart_settings(FILE *fp)
{
  if (comm->me == 0) {
    utils::sfread(FLERR,&temperature,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&cut_global,sizeof(double),1,fp,nullptr,error);
    utils::sfread(FLERR,&seed,sizeof(int),1,fp,nullptr,error);
    utils::sfread(FLERR,&mix_flag,sizeof(int),1,fp,nullptr,error);
  }
  MPI_Bcast(&temperature,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&seed,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);

  // initialize Marsaglia RNG with processor-unique seed
  // same seed that pair_style command initially specified

  if (random) delete random;
  random = new RanMars(lmp,seed + comm->me);
}

/* ---------------------------------------------------------------------- */

double PairDPDfdtExt::single(int /*i*/, int /*j*/, int itype, int jtype, double rsq,
                       double /*factor_coul*/, double factor_dpd, double &fforce)
{
  double r,rinv,wdc,phi;

  r = sqrt(rsq);
  if (r < EPSILON) {
    fforce = 0.0;
    return 0.0;
  }

  rinv = 1.0/r;
  if (r <=cut[itype][jtype]) wdc = 1.0 -r/cut[itype][jtype];
  else wdc=0.0;
  fforce = a0[itype][jtype]*wdc * factor_dpd*rinv;

  phi = 0.5*a0[itype][jtype]*cut[itype][jtype] * wdc*wdc;
  return factor_dpd*phi;
}

