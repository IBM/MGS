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
