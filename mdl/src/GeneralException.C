// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "GeneralException.h"
#include <string>

GeneralException::GeneralException(const std::string& error) 
   : _error(error) 
{

}

const std::string& GeneralException::getError() const
{
   return _error;
}

void GeneralException::setError(const std::string& error) 
{
   _error = error;
}

GeneralException::~GeneralException() 
{

}        

