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

#ifndef C_compCategoryBase_H
#define C_compCategoryBase_H
#include "Mdl.h"

#include "C_interfaceImplementorBase.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_interfacePointerList;
class CompCategoryBase;

class C_compCategoryBase : public C_interfaceImplementorBase {

   public:
      virtual void execute(MdlContext* context);
      C_compCategoryBase();
      C_compCategoryBase(const std::string& name, C_interfacePointerList* ipl
			 , C_generalList* gl);
      virtual void duplicate(std::auto_ptr<C_compCategoryBase>& rv) const;
      virtual ~C_compCategoryBase();
      void executeCompCategoryBase(MdlContext* context,
				   CompCategoryBase* cc) const;
};


#endif // C_compCategoryBase_H
