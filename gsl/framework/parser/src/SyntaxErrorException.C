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
#include "SyntaxErrorException.h"

#include <iostream>

SyntaxErrorException::SyntaxErrorException(std::string errCode, 
					   bool first)
   : _gslErrorCode(errCode), _first(first)
{
}


SyntaxErrorException::SyntaxErrorException(const SyntaxErrorException& l)
   : _gslErrorCode(l._gslErrorCode), _first(l._first)
{
}

SyntaxErrorException& SyntaxErrorException::operator=(
   const SyntaxErrorException& l)
{
   if (this != &l) {
      _gslErrorCode = l._gslErrorCode;
      _first = l._first;
   }
   return *this;
}

std::string SyntaxErrorException::getError()
{
   return(_gslErrorCode + " (SyntaxErrorException)");
}

std::string SyntaxErrorException::what()
{
   return "SyntaxErrorException";
}

void SyntaxErrorException::printError()
{
   if (_gslErrorCode != "") {
      std::cerr << _gslErrorCode << std::endl;      
   }
}

void SyntaxErrorException::resetError()
{
   _gslErrorCode = "";
}
