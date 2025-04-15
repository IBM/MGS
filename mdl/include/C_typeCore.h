// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_typeCore_H
#define C_typeCore_H
#include "Mdl.h"

#include "DataType.h"
#include "C_production.h"
#include <memory>
#include <string>

class MdlContext;

class C_typeCore : public C_production {
   using C_production::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_typeCore();
      C_typeCore(DataType* dt);
      C_typeCore(const std::string& s);
      C_typeCore(const C_typeCore& rv);
      virtual void duplicate(std::unique_ptr<C_typeCore>&& rv) const;
      void releaseDataType(std::unique_ptr<DataType>&& dt);
      virtual ~C_typeCore();
      
   private:
      DataType* _dataType;
      std::string _id;

};


#endif // C_typeCore_H
