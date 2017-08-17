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

#ifndef SYNTAXERROREXCEPTION_H
#define SYNTAXERROREXCEPTION_H
#include "Copyright.h"

#include <string>


class SyntaxErrorException
{
   public:
      SyntaxErrorException(std::string errcode = "", 
			   bool first = false);
      SyntaxErrorException(const SyntaxErrorException& l);
      SyntaxErrorException& operator=(const SyntaxErrorException& l);
      std::string getError();
      std::string what();
      void printError();
      void resetError();
      bool isFirst() { return _first; };
      void setFirst() { _first = true; };
   private:
      std::string _lensErrorCode;
      bool _first;
};
#endif
