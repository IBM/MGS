// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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
