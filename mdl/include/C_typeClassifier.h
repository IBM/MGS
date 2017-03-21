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

#ifndef C_typeClassifier_H
#define C_typeClassifier_H
#include "Mdl.h"

#include "C_production.h"
#include <memory>

class MdlContext;
class C_typeCore;
class C_array;
class DataType;

class C_typeClassifier : public C_production {

   public:
      virtual void execute(MdlContext* context);
      C_typeClassifier();
      C_typeClassifier(C_typeCore* tc, bool pointer = false);
      C_typeClassifier(C_array* a, bool pointer = false);
      C_typeClassifier(const C_typeClassifier& rv);
      virtual void duplicate(std::auto_ptr<C_typeClassifier>& rv) const;
      void releaseDataType(std::auto_ptr<DataType>& dt);
      virtual ~C_typeClassifier();

   private:
      C_typeCore* _typeCore;
      C_array* _array;
      bool _pointer;
      DataType* _dataType;
};


#endif // C_typeClassifier_H
