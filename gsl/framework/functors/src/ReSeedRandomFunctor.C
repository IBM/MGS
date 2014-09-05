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

#include "ReSeedRandomFunctor.h"
#include "FunctorType.h"
#include "FunctorDataItem.h"
#include "NumericDataItem.h"
#include "LensContext.h"
#include "InstanceFactoryQueriable.h"
#include "FloatDataItem.h"
#include "DataItemQueriable.h"
#include "FunctorDataItem.h"
#include "SyntaxErrorException.h"
#include "Simulation.h"
#include "rndm.h"

#include <sstream>
class FunctorType;

void ReSeedRandomFunctor::doInitialize(LensContext *c, 
				       const std::vector<DataItem*>& args)
{
  if (args.size()==0) {
      std::ostringstream msg;
      msg << "ReSeedRandomFunctor: invalid arguments!" << std::endl
	  << "\texpected: ReSeedRandomFunctor(long seed)" << std::endl;
      throw SyntaxErrorException(msg.str());
  }

  NumericDataItem* ndi = dynamic_cast<NumericDataItem*>(args[0]);
  if (ndi == 0) {
    throw SyntaxErrorException(
      "Dynamic Cast of workUnitSeed Numeric Data Item Failed in ReSeedRandomFunctor");
  }
  else
    _seed = ndi->getInt();

  getRangen().reSeed( _seed, c->sim->getRank() ); // reSeed is rank dependent, with RNG in rank n seeded with seed+n
  getSharedRangen().reSeedShared( _seed-1 );      // reSeedShared is not rank dependent; 
                                                  // _seed-1 is necessary so that seed is differen from Rangen on rank 0
  c->sim->getSharedWorkUnitRandomSeedGenerator().reSeedShared( lrandom(getSharedRangen()) );
  c->sim->getWorkUnitRandomSeedGenerator().reSeed( lrandom(getRangen()), c->sim->getRank() );

}


void ReSeedRandomFunctor::doExecute(LensContext *c, 
				    const std::vector<DataItem*>& args, 
				    std::auto_ptr<DataItem>& rvalue)
{
}


void ReSeedRandomFunctor::duplicate(std::auto_ptr<Functor> &fap) const
{
   fap.reset(new ReSeedRandomFunctor(*this));
}


ReSeedRandomFunctor::ReSeedRandomFunctor()
{
}

ReSeedRandomFunctor::~ReSeedRandomFunctor()
{
}
