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

#ifndef C_interfacePointer_H
#define C_interfacePointer_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>
#include <string>

class MdlContext;
class Interface;

class C_interfacePointer : public C_production {

   public:
      virtual void execute(MdlContext* context);
      C_interfacePointer();
      C_interfacePointer(const std::string& name);
      C_interfacePointer(const C_interfacePointer& rv);
      virtual void duplicate(std::auto_ptr<C_interfacePointer>& rv) const;
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
