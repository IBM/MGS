// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BMC-YKT-08-23-2011-2
//
// (C) Copyright IBM Corp. 2005-2014  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef C_DECLARATION_LOAD_PHASES_H
#define C_DECLARATION_LOAD_PHASES_H
#include "Copyright.h"

#include <vector>

#include "C_declaration.h"

class LensContext;
class DataItem;
class C_phase;
class C_phase_list;

class C_declaration_load_phases : public C_declaration
{
   public:
      C_declaration_load_phases(const C_declaration_load_phases&);
      C_declaration_load_phases(C_phase_list *, SyntaxError *);
      virtual ~C_declaration_load_phases();
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      virtual C_declaration* duplicate() const;

   private:
      C_phase_list* _phaseList;
};
#endif
