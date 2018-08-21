// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-07-18-2017
//
// (C) Copyright IBM Corp. 2005-2017  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

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

   public:
      virtual void execute(MdlContext* context);
      C_interfacePointerList();
      C_interfacePointerList(C_interfacePointer* ip);
      C_interfacePointerList(C_interfacePointerList* ipl, 
			     C_interfacePointer* ip);
      C_interfacePointerList(const C_interfacePointerList& rv);
      virtual void duplicate(std::auto_ptr<C_interfacePointerList>& rv) const;
      void releaseInterfaceVec(std::auto_ptr<std::vector<Interface*> >& ipv);
      virtual ~C_interfacePointerList();

   private:
      C_interfacePointer* _interfacePointer;
      C_interfacePointerList* _interfacePointerList;
      std::vector<Interface*>* _interfacePointerVec;
};


#endif // C_interfacePointerList_H
