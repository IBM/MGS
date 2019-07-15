// =================================================================
// Licensed Materials - Property of IBM
//
// "Restricted Materials of IBM"
//
// BCM-YKT-11-14-2018
//
// (C) Copyright IBM Corp. 2005-2018  All rights reserved
//
// US Government Users Restricted Rights -
// Use, duplication or disclosure restricted by
// GSA ADP Schedule Contract with IBM Corp.
//
// =================================================================

#ifndef Constants_H
#define Constants_H
#include "Mdl.h"

#include <map>
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
const std::string PREFIX_PROXY_MEMBERNAME = "proxy_um_";
const std::string SUFFIX_MEMBERNAME_ARRAY = "_start_offset";
const std::string SUFFIX_MEMBERNAME_ARRAY_MAXELEMENTS = "_max_elements";
const std::string REF_CC_OBJECT = "_container"; // data member that references to CompCategory object
const std::string REF_INDEX = "index"; // index to data array as stored in ompCategory object
const std::string REF_DEMARSHALLER_INDEX = "demarshaller_index"; // index to CCDermarshaller 
const std::string STR_GPU_CHECK_START = "#ifdef HAVE_GPU\n";
const std::string GPUCONDITIONAL = "HAVE_GPU";
const std::string STR_GPU_CHECK_END = "#endif\n";
const std::string STR_FPGA_CHECK_START = "#ifdef HAVE_FPGA\n";
const std::string STR_FPGA_CHECK_END = "#endif\n";
const std::string GETCOMPCATEGORY_FUNC_NAME = "getCompCategory";
const std::string SETCOMPCATEGORY_FUNC_NAME = "setCompCategory";
const std::string GETDEMARSHALLERINDEX_FUNC_NAME = "getDemarshallerIndex";
const std::string GETDEMARSHALLER_FUNC_NAME = "getDemarshaller";  // if not found, return 'nullptr'
const std::string FINDDEMARSHALLER_FUNC_NAME = "findDemarshaller"; //create if not existing
const std::string GETDATA_FUNC_NAME = "getDataIndex";

const std::string MEMORY_LOCATION = "Array_Flat<int>::MemLocation::UNIFIED_MEM";
//const std::string MEMORY_LOCATION = "Array_Flat<int>::MemLocation::CPU";
const std::string COMMON_MAX_SUBARRAY_SIZE = "1000"; // the max size for array datamember

const std::string REUSENA_CONDITIONAL = "REUSE_NODEACCESSORS";
const std::string TRACK_SAS_CONDITIONAL = "TRACK_SUBARRAY_SIZE";

enum class MachineType {
   CPU, GPU, FPGA
};

static std::map<MachineType, std::string> MachineTypeNames =
  {
    { MachineType::CPU, "CPU"},
    { MachineType::GPU, "GPU"},
    { MachineType::FPGA, "FPGA"}
  };

template< typename T >
class Enum
{
   //NOTE: functional, but not completed
public:
   class Iterator
   {
   public:
      Iterator( int value ) :
         m_value( value )
      { }

      T operator*( void ) const
      {
         return (T)m_value;
      }

      void operator++( void )
      {
         ++m_value;
      }

      bool operator!=( Iterator rhs )
      {
         return m_value != rhs.m_value;
      }

   private:
      int m_value;
   };

};

template< typename T >
typename Enum<T>::Iterator begin( Enum<T> )
{
   return typename Enum<T>::Iterator( (int)T::First );
}

template< typename T >
typename Enum<T>::Iterator end( Enum<T> )
{
   return typename Enum<T>::Iterator( ((int)T::Last) + 1 );
}

#endif
