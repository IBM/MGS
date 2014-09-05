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

#ifndef C_production_H
#define C_production_H
#include "Mdl.h"

#include "MdlContext.h"
#include <memory>

class C_production {

   public:
      virtual void execute(MdlContext* context);
      C_production();
      C_production(const C_production& rv);
      virtual void duplicate(std::auto_ptr<C_production>& rv) const;
      virtual ~C_production();
      int getLineNumber() const;
      void setLineNumber(int lineNumber);
      const std::string& getFileName() const;
      void setFileName(const std::string& fileName);
      void setTokenLocation(const std::string& fileName, int lineNumber);
   private:
      int _lineNumber;
      std::string _fileName;

};
#endif // C_production_H
