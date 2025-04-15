// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "NumIntNoPhase.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>

void EUNoPhase::initializeIterator(int numvars_, double dT_)
{
  dT = dT_; 
  dx1.increaseSizeTo(numvars_);
}


void EUNoPhase::callIterate(ShallowArray< double > & x)
{
  derivs(x,dx1); 

  for (i1=dx1.begin(),i2=x.begin();i1!=dx1.end();i1++,i2++)
   *i2 = *i2 + dT**i1;

}



void RK4NoPhase::initializeIterator(int numvars_, double dT_)
{
  dT = dT_; dT2=dT_/2.0; dT6 = dT_/6.0;
  dx1.increaseSizeTo(numvars_);
  dx2.increaseSizeTo(numvars_);
  dx3.increaseSizeTo(numvars_);
  x1.increaseSizeTo(numvars_);  
}


void RK4NoPhase::callIterate(ShallowArray< double > & x)
{
  derivs(x,dx1); 

  
  for (i1=x1.begin(),i2=x.begin(),i3=dx1.begin();i1!=x1.end();i1++,i2++,i3++) 
    *i1 = *i2 + dT2**i3;

  
  derivs(x1,dx2);

  
  for (i1=x1.begin(),i2=x.begin(),i3=dx2.begin();i1!=x1.end();i1++,i2++,i3++) 
    *i1 = *i2 + dT2**i3;
  derivs(x1,dx3);
  for (i1=x1.begin(),i2=x.begin(),i3=dx3.begin(),i4=dx2.begin();i1!=x1.end();i1++,i2++,i3++,i4++)
    { 
    *i1 = *i2 + dT**i3;
    *i3 += *i4;
    }
  
  derivs(x1,dx2);
  for (i1=dx1.begin(),i2=x.begin(),i3=dx3.begin(),i4=dx2.begin();i1!=dx1.end();i1++,i2++,i3++,i4++)
   *i2 = *i2 + dT6*(*i1 + *i4 + 2.0**i3);
}
