// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
      virtual void duplicate(std::unique_ptr<C_production>&& rv) const;
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
