// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef C_phase_mapping_H
#define C_phase_mapping_H
#include "Copyright.h"

#include <string>
#include "C_production.h"

class LensContext;
class SyntaxError;

class C_phase_mapping : public C_production
{
   public:
      C_phase_mapping(const C_phase_mapping&);
      C_phase_mapping(const std::string& , const std::string& , 
		      SyntaxError *);
      virtual ~C_phase_mapping();
      virtual C_phase_mapping* duplicate() const;
      virtual void internalExecute(LensContext *);
      virtual void checkChildren();
      virtual void recursivePrint();
      const std::string& getModelPhase() {
	 return _modelPhase;
      }
      const std::string& getSimulationPhase() {
	 return _simulationPhase;
      }

   private:
      std::string _modelPhase;
      std::string _simulationPhase;
};
#endif
