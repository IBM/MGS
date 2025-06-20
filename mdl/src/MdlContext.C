// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "MdlContext.h"
#include "MdlLexer.h"

MdlContext::MdlContext()
   : _lexer(0), _lineNumber(0), _fileName(""), _error(false)
   , _errorDisplayed(false), _lastErrorFileName(""), _lastErrorLine(0) {
}

MdlContext::MdlContext(MdlContext* c)
   : _lexer(c->_lexer), _generatables(c->_generatables)
   , _lineNumber(c->_lineNumber), _fileName(c->_fileName)
   , _error(c->_error), _errorDisplayed(c->_errorDisplayed)
   , _lastErrorFileName(c->_lastErrorFileName)
   , _lastErrorLine(c->_lastErrorLine) {
}

void MdlContext::synchronize() {
   if (_lexer) {
       _lineNumber = _lexer->lineCount;
       _fileName = _lexer->currentFileName;
       
       // Optional: Debug output
       // std::cerr << "Context synchronized: " << _fileName << ":" << _lineNumber << std::endl;
   }
}

std::string const &MdlContext::getFileName() const {
   return _fileName;
}

int MdlContext::getLineNumber() const {
   return  _lineNumber;
}

void MdlContext::setLastError(std::string& fileName, int line) {
   _lastErrorFileName = fileName;
   _lastErrorLine = line;
}

bool MdlContext::isSameErrorLine(std::string& fileName, int line) {
   if ((fileName == _lastErrorFileName) && (line == _lastErrorLine)) {
      return true;
   } else {
      return false;
   }
}

MdlContext::~MdlContext() {
}
