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
      int getExecutionLineNumber() const;
      std::string const & getExecutionFileName() const;
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
