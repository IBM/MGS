// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef MdlContext_H
#define MdlContext_H
#include "Mdl.h"


#include <string>
#include "Generatable.h"
#include "MemberContainer.h"

class MdlLexer;

class MdlContext
{
   public:
      MdlContext();
      MdlContext(MdlContext* c);

      void synchronize();
      int getLineNumber() const;
      std::string const & getFileName() const;
      std::string const & getLastErrorFileName() {return _lastErrorFileName;}  
      int getLastErrorLine() {return _lastErrorLine;}
      void setError() { _error = true;};
      bool isError() { return _error;};
      void setErrorDisplayed(bool errorDisplayed) { 
	      _errorDisplayed = errorDisplayed;
      };
      bool isErrorDisplayed() { return _errorDisplayed;};
      void setLastError(std::string& fileName, int line);
      bool isSameErrorLine(std::string& fileName, int line);
      ~MdlContext();
      
      MdlLexer *_lexer;
      MemberContainer<Generatable> *_generatables;
      
   private:
      int _lineNumber;
      std::string _fileName;
      bool _error;
      bool _errorDisplayed;
      std::string _lastErrorFileName;
      int _lastErrorLine;
};
#endif
