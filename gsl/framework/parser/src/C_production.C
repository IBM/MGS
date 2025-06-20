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
#include "C_production.h"
#include "SyntaxError.h"
#include "SyntaxErrorException.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>

C_production::C_production(const C_production& rv)
   : _error(0) 
{
   if (rv._error) {
      _error = (rv._error)->duplicate();
   }
}

C_production::C_production(SyntaxError* error)
   : _error(error)
{
}

C_production::~C_production()
{
   delete _error;
}


bool C_production::isError() 
{
   if (_error) {
      return _error->isError();
   } else {
      std::cerr << "SEGF: C_production::isError() " << std::endl;
      exit (-1);
   }
   return false;
}

void C_production::setError() 
{
   if (_error) {
      _error->setError(true);
   } else {
      std::cerr << "SEGF: C_production::isError() " << std::endl;
      exit (-1);
   } 
}

void C_production::printErrorMessage() 
{
   if (_error) {
      _error->printMessage();
   } else {
      std::cerr << "SEGF: C_production::printErrorMessage() " << std::endl;
      exit (-1);
   }
}
void C_production::throwError(const std::string &mes)
{
   _error->setOriginal();
   _error->appendMessage(mes);
   setError();
   throw SyntaxErrorException("", true);
}

void C_production::execute(GslContext *c)
{
#if HAVE_MPI
//   int mySpaceId;
//   MPI_Comm_rank(MPI_COMM_WORLD, &mySpaceId);
#endif

   try {
      internalExecute(c);
   } catch (SyntaxErrorException& e) {
// Remove when exceptions are fixed in AIX gcc [begin|sgc]
// #ifdef AIX 
//       e.printError();
//       _error->setOriginal();
//       _error->printMessage();
//       exit(-1);
// #endif
// Remove when exceptions are fixed in AIX gcc [end|sgc]
      if (e.isFirst() == false) {
	 _error->setOriginal();
	 e.setFirst();
      }
      setError();
      throw;
   }
}
