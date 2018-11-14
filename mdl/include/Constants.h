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
const std::string PREFIX_MEMBERNAME = "um_";
const std::string SUFFIX_MEMBERNAME_ARRAY = "_start_offset";
const std::string SUFFIX_MEMBERNAME_ARRAY_MAXELEMENTS = "_max_elements";
const std::string REF_CC_OBJECT = "_container"; // data member that references to CompCategory object
const std::string REF_INDEX = "index"; // index to data array as stored in ompCategory object
const std::string STR_GPU_CHECK_START = "#ifdef HAVE_GPU\n";
const std::string GPUCONDITIONAL = "HAVE_GPU";
const std::string STR_GPU_CHECK_END = "#endif\n";
const std::string GETCOMPCATEGORY_FUNC_NAME = "getContainer";
const std::string SETCOMPCATEGORY_FUNC_NAME = "setCompCategory";
enum class MachineType {
   CPU, GPU
};
#endif
