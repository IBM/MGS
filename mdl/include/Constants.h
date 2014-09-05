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

#ifndef Constants_H
#define Constants_H
#include "Mdl.h"

#include <string>

const std::string TAB = "   ";
const std::string PREFIX = "CG_";
const std::string INATTRPSETNAME = "CG_castedPSet";
const std::string OUTATTRPSETNAME = "CG_castedPSet";
const std::string PREDICATEFUNCTIONPREFIX = "CG_predicate_";
const std::string FOUND = "CG_found";
const std::string FALSEASSERT = "assert(false);\n";
const std::string TRIGGERABLEFUNCTIONNAME = "CG_triggerableFunctionName";
const std::string TRIGGERABLENDPLIST = "CG_triggerableNdpList";
const std::string TRIGGERABLECALLER = "CG_triggerableCaller";
const std::string EVENTTYPE = "TriggerableBase::EventType";
const std::string SERIALRETURN = "TriggerableBase::_SERIAL";
const std::string PARALLELRETURN = "TriggerableBase::_PARALLEL";
const std::string UNALTEREDRETURN = "TriggerableBase::_UNALTERED";
const std::string COMPCATEGORY = "CompCategory";
const std::string COMPCATEGORYBASE = "CompCategoryBase";
const std::string SIMULATION = "_sim";
const std::string PUBLISHABLE = "_publishable";
const std::string DATA = "_data";
const std::string SERVICEREQUESTED = "serviceRequested";
const std::string GETSERVICE = "getService_";
const std::string INTERFACENAME = "interfaceName";
const std::string SUBINTERFACENAME = "subInterfaceName";
const std::string PUBDATATYPE = "void*";
const std::string PUBDATANAME = "data";
const std::string SERVICEDESCRIPTORS = "_serviceDescriptors";
const std::string MPICONDITIONAL = "HAVE_MPI";
const std::string NAMESSET = "namesSet";
const std::string PHASE = "phase";
const std::string SENDFUNCTIONPTRTYPE = "CG_T_SendFunctionPtr";
const std::string GETSENDTYPEFUNCTIONPTRTYPE = "CG_T_GetSendTypeFunctionPtr";
const std::string RECVFUNCTIONPTRTYPE = "CG_T_RecvFunctionPtr";
const std::string OUTPUTSTREAM = "OutputStream";
const std::string RECVTEMPLATES = "CG_recvTemplates";
const std::string SENDTEMPLATES = "CG_sendTemplates";
const std::string GETSENDTYPETEMPLATES = "CG_getSendTypeTemplates";
#endif
