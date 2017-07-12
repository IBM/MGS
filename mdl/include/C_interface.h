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

   public:
      virtual void execute(MdlContext* context);
      C_interface();
      C_interface(const std::string& name, C_dataTypeList* dtl);
      C_interface(const C_interface& rv);
      virtual void duplicate(std::auto_ptr<C_interface>& rv) const;
      virtual ~C_interface();
      
   protected:
      std::string _name;
      Interface* _interface;
      C_dataTypeList* _dataTypeList;

};


#endif // C_interface_H
