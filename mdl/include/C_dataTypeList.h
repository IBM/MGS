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

   public:
      virtual void execute(MdlContext* context);
      C_dataTypeList();
      C_dataTypeList(C_dataType* dt);
      C_dataTypeList(C_dataTypeList* dtl, C_dataType* dt);
      C_dataTypeList(const C_dataTypeList& rv);
      virtual void duplicate(std::auto_ptr<C_dataTypeList>& rv) const;
      void releaseDataTypeVec(std::auto_ptr<std::vector<DataType*> >& dtv);
      virtual ~C_dataTypeList();

   private:
      C_dataType* _dataType;
      C_dataTypeList* _dataTypeList;
      std::vector<DataType*>* _dataTypeVec;

      void deleteVector();
      void deepCopyVector(const C_dataTypeList& rv);
};


#endif // C_dataTypeList_H
