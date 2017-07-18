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

#ifndef ParserClasses_H
#define ParserClasses_H
#include "Mdl.h"

#include "C_array.h"
// #include "C_argumentMapping.h"
#include "C_argumentToMemberMapper.h"
#include "C_compCategoryBase.h"
#include "C_connection.h"
#include "C_connectionCCBase.h"
#include "C_constant.h"
#include "C_dataType.h"
#include "C_dataTypeList.h"
#include "C_edge.h"
#include "C_edgeConnection.h"
#include "C_execute.h"
#include "C_finalPhase.h"
#include "C_functor.h"
#include "C_general.h"
#include "C_generalList.h"
#include "C_identifierList.h"
#include "C_inAttrPSet.h"
#include "C_initialize.h"
#include "C_initPhase.h"
#include "C_interface.h"
#include "C_interfaceImplementorBase.h"
#include "C_interfaceMapping.h"
#include "C_interfacePointer.h"
#include "C_interfacePointerList.h"
#include "C_interfaceToInstance.h"
#include "C_interfaceToShared.h"
#include "C_instanceMapping.h"
#include "C_loadPhase.h"
#include "C_nameComment.h"
#include "C_nameCommentList.h"
#include "C_node.h"
#include "C_noop.h"
#include "C_outAttrPSet.h"
#include "C_phase.h"
#include "C_phaseIdentifier.h"
#include "C_phaseIdentifierList.h"
#include "C_predicateFunction.h"
#include "C_psetMapping.h"
#include "C_psetToInstance.h"
#include "C_psetToShared.h"
#include "C_regularConnection.h"
#include "C_returnType.h"
#include "C_runtimePhase.h"
#include "C_shared.h"
#include "C_sharedMapping.h"
#include "C_sharedCCBase.h"
#include "C_struct.h"
#include "C_triggeredFunction.h"
#include "C_triggeredFunctionInstance.h"
#include "C_triggeredFunctionShared.h"
#include "C_toolBase.h"
#include "C_typeClassifier.h"
#include "C_typeCore.h"
#include "C_userFunction.h"
#include "C_userFunctionCall.h"
#include "C_variable.h"
//#include "C_resource.h"                        // added by Jizhu Lu on 02/06/2006
#include "C_computeTime.h"                     // added by Jizhu Lu on 02/06/2006

#include "Predicate.h"
#include "PSetPredicate.h"
#include "InstancePredicate.h"
#include "SharedPredicate.h"
#include "FunctionPredicate.h"

#endif // ParserClasses_H
