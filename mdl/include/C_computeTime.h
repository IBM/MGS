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

#ifndef C_computeTime_H
#define C_computeTime_H
#include "Mdl.h"

#include "C_general.h"
#include <memory>
#include <string>

class MdlContext;
class C_generalList;
class C_identifierList;

class C_computeTime: public C_general {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_computeTime();
      C_computeTime(C_identifierList* identifierList); 
      C_computeTime(const C_computeTime& rv);
      C_computeTime(double& rv);
      C_computeTime& operator=(const C_computeTime& rv);
      virtual void duplicate(std::auto_ptr<C_computeTime>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_computeTime();
      
   private:
      void copyOwnedHeap(const C_computeTime& rv);
      void destructOwnedHeap();
      C_identifierList* _identifierList;
      double _computeTime;
};


#endif // C_computeTime_H
