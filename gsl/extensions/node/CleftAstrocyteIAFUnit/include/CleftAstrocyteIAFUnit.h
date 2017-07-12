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

#ifndef CleftAstrocyteIAFUnit_H
#define CleftAstrocyteIAFUnit_H

#include "Lens.h"
#include "CG_CleftAstrocyteIAFUnit.h"
#include "rndm.h"

class CleftAstrocyteIAFUnit : public CG_CleftAstrocyteIAFUnit
{
   public:
      void initialize(RNG& rng);
      void update(RNG& rng);
      virtual void setGlutamateIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CleftAstrocyteIAFUnitInAttrPSet* CG_inAttrPset, CG_CleftAstrocyteIAFUnitOutAttrPSet* CG_outAttrPset);
<<<<<<< HEAD
=======
      virtual void setECBIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_CleftAstrocyteIAFUnitInAttrPSet* CG_inAttrPset, CG_CleftAstrocyteIAFUnitOutAttrPSet* CG_outAttrPset);
>>>>>>> origin/team-A
      virtual ~CleftAstrocyteIAFUnit();
};

#endif
