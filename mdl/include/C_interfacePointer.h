// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_interfacePointer_H
#define C_interfacePointer_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>
#include <string>

class MdlContext;
class Interface;

class C_interfacePointer : public C_production {
   using C_production::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_interfacePointer();
      C_interfacePointer(const std::string& name);
      C_interfacePointer(const C_interfacePointer& rv);
      virtual void duplicate(std::unique_ptr<C_interfacePointer>&& rv) const;
      virtual ~C_interfacePointer();
      Interface* getInterface();
      void setInterface(Interface* interface);
      const std::string& getName() const;
      void setName(const std::string& name);
      

   private:
      Interface* _interface;
      std::string _name;
};


#endif // C_interfacePointer_H
