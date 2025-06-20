// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "ConnectionException.h"
#include "GeneralException.h"

ConnectionException::ConnectionException(const std::string& error) 
   : GeneralException(error) 
{

}

ConnectionException::~ConnectionException() 
{

}
