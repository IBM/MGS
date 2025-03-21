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

#ifndef C_dataType_H
#define C_dataType_H
#include "Mdl.h"

#include "C_general.h"
#include "C_nameCommentList.h"
#include <memory>
#include <string>

class MdlContext;
class DataType;
class C_typeClassifier;
class C_generalList;

class C_dataType : public C_general {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_dataType(); 
      C_dataType(C_typeClassifier* tc, C_nameCommentList* ncl, 
		 bool derived = false, bool optional = false); 
      C_dataType(const C_dataType& rv);
      virtual void duplicate(std::unique_ptr<C_dataType>&& rv) const;
      virtual void duplicate(std::unique_ptr<C_general>&& rv) const;
      void releaseDataTypeVec(std::unique_ptr<std::vector<DataType*> >& dtv);
      virtual ~C_dataType();
      
   private:
      bool _derived;
      // _optional is not a property of DataType, it is simply used to select
      // the right list to insert the created DataType into the general list.
      bool _optional;
      C_typeClassifier* _typeClassifier;
      C_nameCommentList* _nameCommentList;
      std::vector<DataType*>* _dataTypeVec;

      void deleteVector();
      void deepCopyVector(const C_dataType& rv);
};


#endif // C_dataType_H
