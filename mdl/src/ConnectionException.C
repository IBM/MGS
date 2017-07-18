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

#include "ConnectionException.h"
#include "GeneralException.h"

ConnectionException::ConnectionException(const std::string& error) 
   : GeneralException(error) 
{

}

ConnectionException::~ConnectionException() 
{

}
