// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "C_production_adi.h"
#include "C_production.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "ArrayDataItem.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

C_production_adi::C_production_adi(const C_production_adi& rv)
   : C_production(rv)
{
}


C_production_adi::C_production_adi(SyntaxError* error)
   : C_production(error)
{
}

C_production_adi::~C_production_adi()
{
}

void C_production_adi::execute(LensContext *c, ArrayDataItem* adi)
{
   try {
      internalExecute(c, adi);
   } catch (SyntaxErrorException& e) {
      if (e.isFirst() == false) {
	 _error->setOriginal();
	 e.setFirst();
      }
      setError();
      throw;
   }
}

void C_production_adi::internalExecute(LensContext *c)
{
   std::cerr << "Internal error, internalExecute of C_production_adi called, exiting..." 
	     << std::endl;
   exit (-1);
}

void C_production_adi::execute(LensContext *c)
{
   C_production::execute(c);
}

