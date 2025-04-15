// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_toolBase_H
#define C_toolBase_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class ToolBase;

class C_toolBase : public C_production {
   protected:
   using C_production::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_toolBase();
      C_toolBase(const std::string& name, C_generalList* gl);
      C_toolBase(const C_toolBase& rv);
      virtual void duplicate(std::unique_ptr<C_toolBase>&& rv) const;
      virtual ~C_toolBase();
      void executeToolBase(MdlContext* context, ToolBase* tb) const;
   protected:
      std::string _name;
      C_generalList* _generalList;     
};


#endif // C_toolBase_H
