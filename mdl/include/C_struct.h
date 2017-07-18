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

#ifndef C_struct_H
#define C_struct_H
#include "Mdl.h"

#include "C_general.h"
#include <memory>
#include <string>

class MdlContext;
class StructType;
class C_dataTypeList;
class C_generalList;

class C_struct : public C_general {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_struct();
      C_struct(C_dataTypeList* dtl);
      C_struct(const std::string& name, C_dataTypeList* dtl, 
	       bool frameWorkElement = false);
      C_struct(const C_struct& rv);
      virtual void duplicate(std::auto_ptr<C_struct>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_struct();
      
   protected:
      std::string _name;
      StructType* _struct;
      C_dataTypeList* _dataTypeList;
      bool _frameWorkElement;
};


#endif // C_struct_H
