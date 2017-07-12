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
