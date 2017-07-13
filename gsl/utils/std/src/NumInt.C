// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "NumInt.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>


void numInt2::initializeIterator(int numvars_, double dT_)
{
  numvars = numvars_;
  dT = dT_; 
  //dx1.increaseSizeTo(numvars_);
  //x.increaseSizeTo(numvars_);
}



void EU2::initializeIterator(int numvars_, double dT_)
{
  numInt2::initializeIterator(numvars_,dT_);
  //deltaT() = dT_; 
  dx1.increaseSizeTo(numvars_);
  x.increaseSizeTo(numvars_);
}


void EU2::callIteratePhase1()
{

  derivs(x,dx1); 

  for (i1=dx1.begin(),i2=x.begin();i1!=dx1.end();i1++,i2++)
    *i2 = *i2 + deltaT()**i1;

}



void RK4Phased::initializeIterator(int numvars_, double dT_)
{
  //numInt::initializeIterator(numvars_,dT_);
  dT = dT_;
  dT2=dT_/2.0; dT6 = dT_/6.0;
  rk1.dx1.increaseSizeTo(numvars_);
  rk1.dx2.increaseSizeTo(numvars_);
  rk1.dx3.increaseSizeTo(numvars_);
  rk1.x1.increaseSizeTo(numvars_);  
  rk1.x.increaseSizeTo(numvars_);  
}




void RK4Phased::callIteratePhase1(RK4PhasedVars & rk1)
{

  derivs(rk1.x,rk1.dx1); 

  //x->dx1
  //dx1,x -> x1

  for (rk1.i1=rk1.x1.begin(),rk1.i2=rk1.x.begin(),rk1.i3=rk1.dx1.begin();rk1.i1!=rk1.x1.end();rk1.i1++,rk1.i2++,rk1.i3++) 
    *rk1.i1 = *rk1.i2 + dT2**rk1.i3;

}

void RK4Phased::callIteratePhase2(RK4PhasedVars & rk1)
{

  derivs(rk1.x1,rk1.dx2);
  //x1 - > dx2
  //x, dx2 -> x1

  for (rk1.i1=rk1.x1.begin(),rk1.i2=rk1.x.begin(),rk1.i3=rk1.dx2.begin();rk1.i1!=rk1.x1.end();rk1.i1++,rk1.i2++,rk1.i3++) 
    *rk1.i1 = *rk1.i2 + dT2**rk1.i3;

}

void RK4Phased::callIteratePhase3(RK4PhasedVars & rk1)
{

  derivs(rk1.x1,rk1.dx3);
  //x1 -> dx3
  //x, dx3, dx2 -> x1

  for (rk1.i1=rk1.x1.begin(),rk1.i2=rk1.x.begin(),rk1.i3=rk1.dx3.begin(),rk1.i4=rk1.dx2.begin();
       rk1.i1!=rk1.x1.end();rk1.i1++,rk1.i2++,rk1.i3++,rk1.i4++)
    { 
      *rk1.i1 = *rk1.i2 + dT**rk1.i3;
      *rk1.i3 += *rk1.i4;
    }

}

void RK4Phased::callIteratePhase4(RK4PhasedVars & rk1)
{
  
  derivs(rk1.x1,rk1.dx2);
  //x1 -> dx2
  //dx1, x, dx3, dx2 -> x

  for (rk1.i1=rk1.dx1.begin(),rk1.i2=rk1.x.begin(),rk1.i3=rk1.dx3.begin(),rk1.i4=rk1.dx2.begin();
       rk1.i1!=rk1.dx1.end();rk1.i1++,rk1.i2++,rk1.i3++,rk1.i4++)
   *rk1.i2 = *rk1.i2 + dT6*(*rk1.i1 + *rk1.i4 + 2.0**rk1.i3);

}


/*

void LypInt::initializeIterator(int numvars_, double dT_)
{

  RK4Phased::initializeIterator(numvars_,dT_);
     
  //deltaT() = dT_;
  d0 = DPREC;

  rk2.dx1.increaseSizeTo(numvars_);
  rk2.dx2.increaseSizeTo(numvars_);
  rk2.dx3.increaseSizeTo(numvars_);
  rk2.x1.increaseSizeTo(numvars_);  
  rk2.x.increaseSizeTo(numvars_);  
}


void LypInt::callIteratePhase1()
{

  RK4Phased::callIteratePhase1();

  derivs(rk2.x,rk2.dx1); 

  //x->dx1
  //dx1,x -> x1

  for (rk2.i1=rk2.x1.begin(),rk2.i2=rk2.x.begin(),rk2.i3=rk2.dx1.begin();rk2.i1!=rk2.x1.end();rk2.i1++,rk2.i2++,rk2.i3++) 
    *rk2.i1 = *rk2.i2 + dT2**rk2.i3;

}

void LypInt::callIteratePhase2()
{

  RK4Phased::callIteratePhase2();

  derivs(rk2.x1,rk2.dx2);
  //x1 - > dx2
  //x, dx2 -> x1

  for (rk2.i1=rk2.x1.begin(),rk2.i2=rk2.x.begin(),rk2.i3=rk2.dx2.begin();rk2.i1!=rk2.x1.end();rk2.i1++,rk2.i2++,rk2.i3++) 
    *rk2.i1 = *rk2.i2 + dT2**rk2.i3;

}

void LypInt::callIteratePhase3()
{

  RK4Phased::callIteratePhase3();

  derivs(rk2.x1,rk2.dx3);
  //x1 -> dx3
  //x, dx3, dx2 -> x1

  for (rk2.i1=rk2.x1.begin(),rk2.i2=rk2.x.begin(),rk2.i3=rk2.dx3.begin(),rk2.i4=rk2.dx2.begin();
       rk2.i1!=rk2.x1.end();rk2.i1++,rk2.i2++,rk2.i3++,rk2.i4++)
    { 
      *rk2.i1 = *rk2.i2 + deltaT()**rk2.i3;
      *rk2.i3 += *rk2.i4;
    }

}

void LypInt::callIteratePhase4()
{
  
 RK4Phased::callIteratePhase4();

  derivs(rk2.x1,rk2.dx2);
  //x1 -> dx2
  //dx1, x, dx3, dx2 -> x

  for (rk2.i1=rk2.dx1.begin(),rk2.i2=rk2.x.begin(),rk2.i3=rk2.dx3.begin(),rk2.i4=rk2.dx2.begin();
       rk2.i1!=rk2.dx1.end();rk2.i1++,rk2.i2++,rk2.i3++,rk2.i4++)
   *rk2.i2 = *rk2.i2 + dT6*(*rk2.i1 + *rk2.i4 + 2.0**rk2.i3);

}

*/


