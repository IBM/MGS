// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifdef HAVE_MPI
#include <mpi.h>
#endif
#include "SyntaxError.h"

#include <iostream>
#include <sstream>
#include <cstring>

// Turn on or off the full trace printing
const bool FULLTRACE = false;

SyntaxError::SyntaxError()
   : _errorMessage(""), _error(false)
{
}

SyntaxError::SyntaxError(std::string& fileName, int lineNum, const char* prod,
			 const char* rule, const char* errMsg, bool err)
   : _error(err), _original(err)
{
   std::ostringstream os;
   os << "File " << fileName << ":" << lineNum << ": While parsing " << prod;
   if (strcmp(rule, "") != 0) {
      os << ", using " << rule;
   }
   if (strcmp(errMsg, "") != 0) {
      os << ", " << errMsg;
   }
   _errorMessage = os.str();
}

SyntaxError::SyntaxError(SyntaxError *c)
   : _errorMessage(c->_errorMessage), _error(c->_error), _original(c->_original)
{
}

SyntaxError* SyntaxError::duplicate()
{
   return new SyntaxError(this);
}


SyntaxError::~SyntaxError()
{
}


bool SyntaxError::isError() 
{
   return _error;
}

void SyntaxError::setError(bool error) 
{
   _error = error;
}

void SyntaxError::printMessage() 
{
   if ((FULLTRACE || _original) && _error) {
      std::cerr << _errorMessage << std::endl;
   }
}

void SyntaxError::appendMessage(const std::string& err)
{
   _errorMessage += ", ";
   _errorMessage += err;
}
