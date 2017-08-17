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

#ifndef C_connectionCCBase_H
#define C_connectionCCBase_H
#include "Mdl.h"

#include "C_compCategoryBase.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_interfacePointerList;
class ConnectionCCBase;

class C_connectionCCBase : public C_compCategoryBase {

   public:
      virtual void execute(MdlContext* context);
      C_connectionCCBase();
      C_connectionCCBase(const std::string& name, C_interfacePointerList* ipl
			 , C_generalList* gl);
      C_connectionCCBase(const C_connectionCCBase& rv);
      virtual void duplicate(std::auto_ptr<C_compCategoryBase>& rv) const;
      virtual void duplicate(std::auto_ptr<C_connectionCCBase>& rv) const;
      virtual ~C_connectionCCBase();
      void executeConnectionCCBase(MdlContext* context,
				   ConnectionCCBase* cc) const;

};


#endif // C_connectionCCBase_H
