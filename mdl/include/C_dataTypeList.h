// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_dataTypeList_H
#define C_dataTypeList_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>
#include <vector>

class MdlContext;
class C_dataType;
class DataType;

class C_dataTypeList : public C_production {
   using C_production::duplicate;  // Make base class method visible
   public:
      virtual void execute(MdlContext* context);
      C_dataTypeList();
      C_dataTypeList(C_dataType* dt);
      C_dataTypeList(C_dataTypeList* dtl, C_dataType* dt);
      C_dataTypeList(const C_dataTypeList& rv);
      virtual void duplicate(std::unique_ptr<C_dataTypeList>&& rv) const;
      void releaseDataTypeVec(std::unique_ptr<std::vector<DataType*> >& dtv);
      virtual ~C_dataTypeList();

   private:
      C_dataType* _dataType;
      C_dataTypeList* _dataTypeList;
      std::vector<DataType*>* _dataTypeVec;

      void deleteVector();
      void deepCopyVector(const C_dataTypeList& rv);
};


#endif // C_dataTypeList_H
