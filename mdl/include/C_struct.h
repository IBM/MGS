// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
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
   protected:
      using C_general::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_struct();
      C_struct(C_dataTypeList* dtl);
      C_struct(const std::string& name, C_dataTypeList* dtl, 
	       bool frameWorkElement = false);
      C_struct(const C_struct& rv);
      virtual void duplicate(std::unique_ptr<C_struct>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      virtual ~C_struct();
      
   protected:
      std::string _name;
      StructType* _struct;
      C_dataTypeList* _dataTypeList;
      bool _frameWorkElement;
};


#endif // C_struct_H
