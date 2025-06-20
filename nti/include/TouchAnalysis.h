// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef TOUCHANALYSIS_H
#define TOUCHANALYSIS_H

#include <mpi.h>
#include "Translation.h"
#include "Rotation.h"

#include <vector>
#include <list>

class TouchTable;

class TouchAnalysis
{
	public:
		TouchAnalysis() {}
		std::vector<TouchTable*> & getTouchTables() {return _touchTables;}
		bool virtual analyze(std::list<Translation>& translations, std::list<Rotation>& rotations)=0;
		virtual ~TouchAnalysis() {}
		
	protected:
		std::vector<TouchTable*> _touchTables;
};

#endif
