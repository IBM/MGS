// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_interfacePointerList_H
#define C_interfacePointerList_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>
#include <vector>

class MdlContext;
class C_interfacePointer;
class Interface;

class C_interfacePointerList : public C_production {
   using C_production::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_interfacePointerList();
      C_interfacePointerList(C_interfacePointer* ip);
      C_interfacePointerList(C_interfacePointerList* ipl, 
			     C_interfacePointer* ip);
      C_interfacePointerList(const C_interfacePointerList& rv);
      virtual void duplicate(std::unique_ptr<C_interfacePointerList>&& rv) const;
      void releaseInterfaceVec(std::unique_ptr<std::vector<Interface*> >& ipv);
      virtual ~C_interfacePointerList();

   private:
      C_interfacePointer* _interfacePointer;
      C_interfacePointerList* _interfacePointerList;
      std::vector<Interface*>* _interfacePointerVec;
};


#endif // C_interfacePointerList_H
