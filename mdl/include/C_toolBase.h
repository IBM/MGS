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

   public:
      virtual void execute(MdlContext* context);
      C_toolBase();
      C_toolBase(const std::string& name, C_generalList* gl);
      C_toolBase(const C_toolBase& rv);
      virtual void duplicate(std::auto_ptr<C_toolBase>& rv) const;
      virtual ~C_toolBase();
      void executeToolBase(MdlContext* context, ToolBase* tb) const;
   protected:
      std::string _name;
      C_generalList* _generalList;     
};


#endif // C_toolBase_H
