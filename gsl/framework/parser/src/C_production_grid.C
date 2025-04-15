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
#include "C_production_grid.h"
#include "C_production.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include "Grid.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

C_production_grid::C_production_grid(const C_production_grid& rv)
   : C_production(rv)
{
}

C_production_grid::C_production_grid(SyntaxError* error)
   : C_production(error)
{
}

C_production_grid::~C_production_grid()
{
}

void C_production_grid::execute(LensContext *c, Grid* g)
{
   try {
      internalExecute(c, g);
   } catch (SyntaxErrorException& e) {
      if (e.isFirst() == false) {
	 _error->setOriginal();
	 e.setFirst();
      }
      setError();
      throw;
   }

}

void C_production_grid::internalExecute(LensContext *c)
{
   std::cerr << "Internal error, internalExecute of C_production_grid called, exiting..." 
	     << std::endl;
   exit (-1);
}
