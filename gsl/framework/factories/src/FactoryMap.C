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

#ifdef AIX
// Workaround of a gcc compiler bug in AIX.
#include "FactoryMap.h"
#include "ConstantType.h"
#include "NodeType.h"
#include "EdgeType.h"
#include "FunctorType.h"
#include "TriggerType.h"
#include "StructType.h"
#include "VariableType.h"

FactoryMap<ConstantType>* _GlobalConstantTypeFactoryMap = 0;
FactoryMap<NodeType>* _GlobalNodeTypeFactoryMap = 0;
FactoryMap<EdgeType>* _GlobalEdgeTypeFactoryMap = 0;
FactoryMap<FunctorType>* _GlobalFunctorTypeFactoryMap = 0;
FactoryMap<TriggerType>* _GlobalTriggerTypeFactoryMap = 0;
FactoryMap<StructType>* _GlobalStructTypeFactoryMap = 0;
FactoryMap<VariableType>* _GlobalVariableTypeFactoryMap = 0;


#endif // AIX workaround
