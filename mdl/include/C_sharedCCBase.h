// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-19-2015
//
// (C) Copyright IBM Corp. 2005-2015  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_sharedCCBase_H
#define C_sharedCCBase_H
#include "Mdl.h"

#include "C_connectionCCBase.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_interfacePointerList;
class SharedCCBase;

class C_sharedCCBase : public C_connectionCCBase {

   public:
      virtual void execute(MdlContext* context);
      C_sharedCCBase();
      C_sharedCCBase(const std::string& name, C_interfacePointerList* ipl,
		     C_generalList* gl);
      C_sharedCCBase(const C_sharedCCBase& rv);
      virtual void duplicate(std::auto_ptr<C_compCategoryBase>& rv) const;
      virtual void duplicate(std::auto_ptr<C_sharedCCBase>& rv) const;
      virtual ~C_sharedCCBase();
      void executeSharedCCBase(MdlContext* context, SharedCCBase* cc) const;

};


#endif // C_sharedCCBase_H
