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

void C_production_adi::execute(GslContext *c, ArrayDataItem* adi)
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

void C_production_adi::internalExecute(GslContext *c)
{
   std::cerr << "Internal error, internalExecute of C_production_adi called, exiting..." 
	     << std::endl;
   exit (-1);
}

void C_production_adi::execute(GslContext *c)
{
   C_production::execute(c);
}

