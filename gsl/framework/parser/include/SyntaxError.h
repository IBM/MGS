// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SyntaxError_H
#define SyntaxError_H
#include "Copyright.h"

#include <string>


class LensContext;
class SyntaxError;

class SyntaxError
{
   public:
      SyntaxError();
      SyntaxError(std::string& fileName, int lineNum, const char* prod, 
		  const char* rule = "", const char* errMes = "", bool err=false);
      SyntaxError(SyntaxError *);
      virtual ~SyntaxError ();
      virtual SyntaxError* duplicate();
      bool isError();
      void setError(bool error);
      void printMessage();
      void appendMessage(const std::string& err);
      void setOriginal() { _original = true; };

   private:
      std::string _errorMessage;
      bool _error;
      bool _original;
};
#endif
