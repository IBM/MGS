#include "Lens.h"
#include "GetPreNodeIndex.h"
#include "CG_GetPreNodeIndexBase.h"
#include "LensContext.h"
#include "NodeDescriptor.h"
#include <memory>

void GetPreNodeIndex::userInitialize(LensContext* CG_c) 
{
}

int GetPreNodeIndex::userExecute(LensContext* CG_c) 
{
  return CG_c->connectionContext->sourceNode->getNodeIndex();
}

GetPreNodeIndex::GetPreNodeIndex() 
   : CG_GetPreNodeIndexBase()
{
}

GetPreNodeIndex::~GetPreNodeIndex() 
{
}

void GetPreNodeIndex::duplicate(std::unique_ptr<GetPreNodeIndex>&& dup) const
{
   dup.reset(new GetPreNodeIndex(*this));
}

void GetPreNodeIndex::duplicate(std::unique_ptr<Functor>&& dup) const
{
   dup.reset(new GetPreNodeIndex(*this));
}

void GetPreNodeIndex::duplicate(std::unique_ptr<CG_GetPreNodeIndexBase>&& dup) const
{
   dup.reset(new GetPreNodeIndex(*this));
}

