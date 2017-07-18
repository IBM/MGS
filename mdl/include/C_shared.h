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

#ifndef C_shared_H
#define C_shared_H
#include "Mdl.h"

#include "C_general.h"
#include "Phase.h"
#include "TriggeredFunction.h"
#include <memory>
#include <string>
#include <vector>

class MdlContext;
class C_generalList;
class DataType;

class C_shared : public C_general {

   public:
      virtual void execute(MdlContext* context);
      virtual void addToList(C_generalList* gl);
      C_shared();
      C_shared(C_generalList* generalList); 
      C_shared(const C_shared& rv);
      virtual void duplicate(std::auto_ptr<C_shared>& rv) const;
      virtual void duplicate(std::auto_ptr<C_general>& rv) const;
      virtual ~C_shared();
      void setGeneral(C_general* general);

      void releasePhases(std::auto_ptr<std::vector<Phase*> >& phases);
      void releaseTriggeredFunctions(
	 std::auto_ptr<std::vector<TriggeredFunction*> >& triggeredFunctions);
      void releaseDataTypeVec(std::auto_ptr<std::vector<DataType*> >& dtv);
      void releaseOptionalDataTypeVec(
	 std::auto_ptr<std::vector<DataType*> >& dtv);

      std::vector<Phase*>* getPhases();
      std::vector<TriggeredFunction*>* getTriggeredFunctions();
      std::vector<DataType*>* getDataTypeVec();
      std::vector<DataType*>* getOptionalDataTypeVec();
      
   private:
      C_generalList* _generalList;
      C_general* _general;
};


#endif // C_shared_H
