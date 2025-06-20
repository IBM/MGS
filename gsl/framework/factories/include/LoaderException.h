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
#include "Copyright.h"
// describe the nature of the exception. For
// example, it might contain info about the file
// or symbol name that could not be opened or
// loaded.

#ifndef LOADEREXCEPTION_H
#define LOADEREXCEPTION_H

#include <string>


class LoaderException
{
   private:
      std::string _errCode;
   public:
      LoaderException(std::string errcode = "");
      LoaderException(const LoaderException& l);
      std::string getError();
};
#endif
