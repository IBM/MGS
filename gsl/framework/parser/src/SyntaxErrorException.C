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
#include "SyntaxErrorException.h"

#include <iostream>

SyntaxErrorException::SyntaxErrorException(std::string errCode, 
					   bool first)
   : _lensErrorCode(errCode), _first(first)
{
}


SyntaxErrorException::SyntaxErrorException(const SyntaxErrorException& l)
   : _lensErrorCode(l._lensErrorCode), _first(l._first)
{
}

SyntaxErrorException& SyntaxErrorException::operator=(
   const SyntaxErrorException& l)
{
   if (this != &l) {
      _lensErrorCode = l._lensErrorCode;
      _first = l._first;
   }
   return *this;
}

std::string SyntaxErrorException::getError()
{
   return(_lensErrorCode + " (SyntaxErrorException)");
}

std::string SyntaxErrorException::what()
{
   return "SyntaxErrorException";
}

void SyntaxErrorException::printError()
{
   if (_lensErrorCode != "") {
      std::cerr << _lensErrorCode << std::endl;      
   }
}

void SyntaxErrorException::resetError()
{
   _lensErrorCode = "";
}
