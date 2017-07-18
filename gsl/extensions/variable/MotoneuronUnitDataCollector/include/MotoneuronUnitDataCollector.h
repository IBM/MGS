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

#ifndef MotoneuronUnitDataCollector_H
#define MotoneuronUnitDataCollector_H

#include "Lens.h"
#include "CG_MotoneuronUnitDataCollector.h"
#include <memory>
#include <fstream>
#include <iostream>

#define saveSimple 0

class MotoneuronUnitDataCollector : public CG_MotoneuronUnitDataCollector
{
 public:
  void initialize(RNG& rng);
  void finalize(RNG& rng);
  virtual void dataCollection(Trigger* trigger, NDPairList* ndPairList);
  virtual void getNodeIndices(const String& CG_direction, const String& CG_component, NodeDescriptor* CG_node, Edge* CG_edge, VariableDescriptor* CG_variable, Constant* CG_constant, CG_MotoneuronUnitDataCollectorInAttrPSet* CG_inAttrPset, CG_MotoneuronUnitDataCollectorOutAttrPSet* CG_outAttrPset);
  MotoneuronUnitDataCollector();
  virtual ~MotoneuronUnitDataCollector();
  virtual void duplicate(std::auto_ptr<MotoneuronUnitDataCollector>& dup) const;
  virtual void duplicate(std::auto_ptr<Variable>& dup) const;
  virtual void duplicate(std::auto_ptr<CG_MotoneuronUnitDataCollector>& dup) const;
 private:
  std::ofstream* d_V_m_last_file;
  std::ofstream* d_I_in_file;
  std::ofstream* s_V_m_file;
  std::ofstream* s_I_in_file;
  std::ofstream* i_V_m_file;
  std::ofstream* i_I_in_file;
  /*
  std::ofstream* a_V_m_node_last_file;
  std::ofstream* a_I_in_file;
  */
#if saveSimple == 0
  std::ofstream* d_everythingElse_file;
  std::ofstream* s_everythingElse_file;
  std::ofstream* i_everythingElse_file;
  //  std::ofstream* a_everythingElse_file;
#endif // saveSimple
};

#endif
