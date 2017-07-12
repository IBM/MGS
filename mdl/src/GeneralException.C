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

