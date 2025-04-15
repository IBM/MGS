// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
