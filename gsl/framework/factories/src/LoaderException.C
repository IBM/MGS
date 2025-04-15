// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
