// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "C_production.h"
#include <memory>

void C_production::execute(MdlContext* context)
{

}

C_production::C_production() 
   : _lineNumber(0), _fileName("")
{

}

C_production::C_production(const C_production& rv)
   : _lineNumber(rv._lineNumber), _fileName(rv._fileName)
{
}

void C_production::duplicate(std::unique_ptr<C_production>&& rv)const
{
   rv.reset(new C_production(*this));
}

int C_production::getLineNumber() const
{
   return _lineNumber;
}

void C_production::setLineNumber(int lineNumber)
{
   _lineNumber = lineNumber;
}

const std::string& C_production::getFileName() const
{
   return _fileName;
}

void C_production::setFileName(const std::string& fileName)
{
   _fileName = fileName;
}

void C_production::setTokenLocation(const std::string& fileName, 
				    int lineNumber)
{
   _fileName = fileName;
   _lineNumber = lineNumber;
}

C_production::~C_production() 
{
}


