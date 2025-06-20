// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_interface_H
#define C_interface_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>
#include <string>

class MdlContext;
class Interface;
class C_dataTypeList;

class C_interface : public C_production {
   using C_production::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_interface();
      C_interface(const std::string& name, C_dataTypeList* dtl);
      C_interface(const C_interface& rv);
      virtual void duplicate(std::unique_ptr<C_interface>&& rv) const;
      virtual ~C_interface();
      
   protected:
      std::string _name;
      Interface* _interface;
      C_dataTypeList* _dataTypeList;

};


#endif // C_interface_H
