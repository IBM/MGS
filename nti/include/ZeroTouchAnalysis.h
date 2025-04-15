// =============================================================================
// (C) Copyright IBM Corp. 2005-2025. All rights reserved.
//
// Distributed under the terms of the Apache License
// Version 2.0, January 2004.
// (See accompanying file LICENSE or copy at http://www.apache.org/licenses/.)
//
// =============================================================================
#ifndef ZEROTOUCHANALYSIS_H
#define ZEROTOUCHANALYSIS_H

#include <mpi.h>
#include "TouchAnalysis.h"
#include "Translation.h"
#include "Rotation.h"

class TouchTable;
class Tissue;
class Neuron;

class ZeroTouchAnalysis : public TouchAnalysis
{
	public:
		ZeroTouchAnalysis(Tissue* tissue);
		bool analyze(std::list<Translation>& translations, std::list<Rotation>& rotations);
		~ZeroTouchAnalysis();						

	private:
		Tissue* _tissue;
		Neuron* _neuronPtr;
		Neuron* _neuronEnd;
};

#endif
