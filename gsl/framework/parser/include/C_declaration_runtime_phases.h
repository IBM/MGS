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

#ifndef C_DECLARATION_RUNTIME_PHASES_H
#define C_DECLARATION_RUNTIME_PHASES_H
#include "Copyright.h"

#include <vector>

#include "C_declaration.h"

class LensContext;
class DataItem;
class C_phase;
class C_phase_list;

class C_declaration_runtime_phases : public C_declaration
{
   public:
      C_declaration_runtime_phases(const C_declaration_runtime_phases&);
      C_declaration_runtime_phases(C_phase_list *, SyntaxError *);
      virtual ~C_declaration_runtime_phases();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      virtual C_declaration* duplicate() const;

   private:
      C_phase_list* _phaseList;
};
#endif
