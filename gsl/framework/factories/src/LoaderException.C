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

// Exception class used by the Loader class. The
// optional argument "errcode" can be used to
// describe the nature of the exception. For
// example, it might contain info about the file
// or symbol name that could not be opened or
// loaded.

#include "LoaderException.h"

LoaderException:: LoaderException(std::string errCode)
:_errCode(errCode)
{
}


LoaderException::LoaderException(const LoaderException& l)
:_errCode(l._errCode)
{
}


std::string LoaderException:: getError()
{
   return(_errCode + " (LoaderException)");
}
