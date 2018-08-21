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
