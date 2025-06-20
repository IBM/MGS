// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef SyntaxErrorException_H
#define SyntaxErrorException_H
#include "Mdl.h"

#include "GeneralException.h"

class SyntaxErrorException : public GeneralException {
   public:
      SyntaxErrorException(const std::string& error, 
			   const std::string& fileName = "",
			   int lineNumber = 0);
      ~SyntaxErrorException();        
      bool isCaught() const;
      void setCaught(bool caught = true);
      int getLineNumber() const;
      void setLineNumber(int lineNumber);
      const std::string& getFileName() const;
      void setFileName(const std::string& fileName);
   protected:
      bool _caught;
      int _lineNumber;
      std::string _fileName;

};

#endif // SyntaxErrorException_H
