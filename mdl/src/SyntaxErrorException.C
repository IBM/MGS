// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#include "SyntaxErrorException.h"
#include "GeneralException.h"

SyntaxErrorException::SyntaxErrorException(const std::string& error
					   , const std::string& fileName
					   , int lineNumber) 
   : GeneralException(error), _caught(false), _lineNumber(lineNumber)
   , _fileName(fileName)
{

}

bool SyntaxErrorException::isCaught() const
{
   return _caught;
}

void SyntaxErrorException::setCaught(bool caught)
{
   _caught = caught;
}

int SyntaxErrorException::getLineNumber() const
{
   return _lineNumber;
}

void SyntaxErrorException::setLineNumber(int lineNumber)
{
   _lineNumber = lineNumber;
}

const std::string& SyntaxErrorException::getFileName() const
{
   return _fileName;
}

void SyntaxErrorException::setFileName(const std::string& fileName)
{
   _fileName = fileName;
}

SyntaxErrorException::~SyntaxErrorException() 
{

}
