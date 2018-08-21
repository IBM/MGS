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

#include "C_interfacePointerList.h"
#include "MdlContext.h"
#include "C_interfacePointer.h"
#include "Interface.h"
#include "InternalException.h"
#include <memory>
#include <vector>

void C_interfacePointerList::execute(MdlContext* context) 
{
   if (_interfacePointer == 0) {
      throw InternalException(
	 "_interfacePointer is 0 in C_interfacePointerList::execute");
   }
   _interfacePointer->execute(context);
   if (_interfacePointerList) {
      _interfacePointerList->execute(context);
      delete _interfacePointerVec;
      std::auto_ptr<std::vector<Interface*> > dtv;
      _interfacePointerList->releaseInterfaceVec(dtv);
      _interfacePointerVec = dtv.release();      
   } else {
      _interfacePointerVec = new std::vector<Interface*>();
   }
   _interfacePointerVec->push_back(_interfacePointer->getInterface());
}

C_interfacePointerList::C_interfacePointerList() 
   : C_production(), _interfacePointer(0), _interfacePointerList(0)
   , _interfacePointerVec(0) 
{

}

C_interfacePointerList::C_interfacePointerList(C_interfacePointer* dt) 
   : C_production(), _interfacePointer(dt), _interfacePointerList(0)
   , _interfacePointerVec(0) 
{

}

C_interfacePointerList::C_interfacePointerList(C_interfacePointerList* dtl, 
					       C_interfacePointer* dt) 
   : C_production(), _interfacePointer(dt), _interfacePointerList(dtl)
   , _interfacePointerVec(0) {

}

C_interfacePointerList::C_interfacePointerList(
   const C_interfacePointerList& rv) 
   : C_production(rv), _interfacePointer(0), _interfacePointerList(0)
   , _interfacePointerVec(rv._interfacePointerVec) 
{
   if (rv._interfacePointer) {
      std::auto_ptr<C_interfacePointer> dup;
      rv._interfacePointer->duplicate(dup);
      _interfacePointer = dup.release();
   }
   if (rv._interfacePointerList) {
      std::auto_ptr<C_interfacePointerList> dup;
      rv._interfacePointerList->duplicate(dup);
      _interfacePointerList = dup.release();
   }
}

void C_interfacePointerList::duplicate(
   std::auto_ptr<C_interfacePointerList>& rv) const
{
   rv.reset(new C_interfacePointerList(*this));
}

void C_interfacePointerList::releaseInterfaceVec(
   std::auto_ptr<std::vector<Interface*> >& dtv) 
{
   dtv.reset(_interfacePointerVec);
   _interfacePointerVec = 0;
}

C_interfacePointerList::~C_interfacePointerList() 
{
   delete _interfacePointer;
   delete _interfacePointerList;
   delete _interfacePointerVec;
}
