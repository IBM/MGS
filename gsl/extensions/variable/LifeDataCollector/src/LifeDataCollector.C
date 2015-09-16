#include "Lens.h"
#include "LifeDataCollector.h"
#include "CG_LifeDataCollector.h"
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>

void LifeDataCollector::initialize(RNG& rng) 
{
   // Create the output file...
  file=new std::ofstream(fileName.c_str());
}

void LifeDataCollector::finalize(RNG& rng) 
{
  file->close();
}

void LifeDataCollector::dataCollection(Trigger* trigger, NDPairList* ndPairList) 
{
  ShallowArray<int*>::iterator iter=vals.begin(), end=vals.end();
  for (; iter!=end; ++iter) {
    (*file)<<**iter<<std::endl;
  }
}

LifeDataCollector::LifeDataCollector() 
  : CG_LifeDataCollector(), file(0)
{
}

LifeDataCollector::~LifeDataCollector() 
{
  delete file;
}

void LifeDataCollector::duplicate(std::auto_ptr<LifeDataCollector>& dup) const
{
   dup.reset(new LifeDataCollector(*this));
}

void LifeDataCollector::duplicate(std::auto_ptr<Variable>& dup) const
{
   dup.reset(new LifeDataCollector(*this));
}

void LifeDataCollector::duplicate(std::auto_ptr<CG_LifeDataCollector>& dup) const
{
   dup.reset(new LifeDataCollector(*this));
}

