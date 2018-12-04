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

#ifndef DataType_H
#define DataType_H
#include "Mdl.h"

#include "Constants.h"
#include <string>
#include <memory>
#include <vector>
#include <set>
#include <cassert>

class Class;

class DataType {
   public:
      // Standard functions
      DataType();
      virtual void duplicate(std::auto_ptr<DataType>& rv) const =0;
      virtual ~DataType();        

      // Getter & Setter functions    
      bool isPointer() const {
	 return _pointer;
      }
      virtual void setPointer(bool pointer) {
	 _pointer = pointer;
      }
      bool isDerived() const {
	 return _derived;
      }
      void setDerived(bool derived) {
	 _derived = derived;
      }
      bool isShared() const {
	 return _shared;
      }
      void setShared(bool shared) {
	 _shared = shared;
      }
      const std::string& getName(MachineType mach_type=MachineType::CPU) const {
	 if (mach_type == MachineType::CPU)
	    return _name;
	 else if (mach_type == MachineType::GPU)
	    return _name_gpu; 
	 else 
	    assert(0);
      }
      const std::string& getNameRaw(MachineType mach_type=MachineType::CPU) const {
	 /* use this to access data directly the '_container' from CG_LifeNode */
	 if (mach_type == MachineType::CPU)
	    return _name;
	 else if (mach_type == MachineType::GPU)
	 {
	    return _name_gpu_raw;
	 }
	 else 
	    assert(0);
      }
      void setName(const std::string& name) {
	 _name = name;
	 //_name_gpu = REF_CC_OBJECT+"->" + PREFIX_MEMBERNAME + _name + "[" + REF_INDEX + "]";
	 _name_gpu = GETCOMPCATEGORY_FUNC_NAME+"()->" + PREFIX_MEMBERNAME + _name + "[" + REF_INDEX + "]";
	 _name_gpu_raw = REF_CC_OBJECT+"->" + PREFIX_MEMBERNAME + _name + "[" + REF_INDEX + "]";
      }
      const std::string& getComment() const {
	 return _comment;
      }
      void setComment(const std::string& comment) {
	 _comment = comment;
      }

      virtual bool isBasic() const {
	 return false;
      }
      
      virtual bool isTemplateMarshalled() const {
	 return true;
      }

      virtual bool isTemplateDemarshalled() const {
	 return true;
      }

      virtual bool isSuitableForInterface() const {
	 return true;
      }
      
      virtual bool isArray() const {
	 return false;
      }

      // This function is used to collect all sub datatype names within a struct or
      // array data type which are themselves structs. It is used to ensure that 
      // Struct demarshallers are included within the appropriate proxy demarshaller
      // header file *before* the standard DemarshallerInstances. Without this order
      // the templatized demarshaller of struct will not compile and/or run properly
      virtual void getSubStructDescriptors(std::set<std::string>& subStructTypes) const {}

      // This function is used for displaying the dataType in the screen right
      // after the program is executed, it serves as a feedback to the user.
      std::string getString() const;

      // This function returns the type of the dataType and a * if the 
      // dataType is a pointer.
      std::string getTypeString() const;

      // This function returns the type of the dataType, 
      // e.g., int for an integer.
      virtual std::string getDescriptor() const=0;

      // This function returns the type of the dataType, ensuring the first
      // letter is capital.  It is mainly used for dataItems, it has to be
      // virtual, we can not simply convert  the first character to capital 
      // because os some dataTypes. LongDouble is an example.
      virtual std::string getCapitalDescriptor() const;

      // This function returns the necessary header if this dataType is used in
      // some code. That is, if this dataType is a NodeType, NodeType.h has to
      // be included as a header file, this function returns that string.
      // Also if the dataType is an array dataType, the array type will be
      // inserted to the vector arrayTypeVec.
      virtual std::string getHeaderString(
	 std::vector<std::string>& arrayTypeVec) const;

      // This function returns the necessary header of the dataItem of this
      // dataType if this  dataType is used in some code. For example
      // NodeTypeDataItem,h would be the return value for a NodeType DataType.
      virtual std::string getHeaderDataItemString() const;

      // This function returns the code for initializing this dataType using a
      // DataItem. The DataItem is first type-checked and a exception is 
      // thrown if it does not match the DataType, if it does, the dataType 
      // is initialized by calling the apropriate function on the DataItem.
      virtual std::string getInitializerString(
	 const std::string& diArg, int level = 0,
	 bool isIterator = true, bool forPSet = false) const;

      // This function returns the code for iniitializing this dataType
      // using a DataItem for PSets. It uses getInitializerString internally.
      std::string getPSetString(const std::string& diArg, 
				bool first = false) const;

      // This function produces the checking code, and updates the name to
      // the type checked struct. This function does nothing, it is overriden
      // in StructType. The overriden function is going to change name, that 
      // is why it is not const.
      // !!!Important due to a bug in gcc 3.2.3 in Linux, the first 
      // parameter's change effects previously generated code. this way 
      // is safe. Under normal circumstances the 1st parameter should not be 
      // const and return the new value.
      virtual std::string checkIfStruct(const std::string& name, 
					const std::string& tab,
					std::string& newName) const;

      // This function duplicates the incoming pointer and creates an auto_ptr
      // that has the  value it modifies the name so that it shows the 
      // auto_ptr's name.
      std::string duplicateIfOwned(const std::string& name, 
				   const std::string& tab,
				   std::string& newName) const;

      // This function returns generated code. The returned value is used
      // in the right hand side of the initialization of the DataType.
      virtual std::string getDataFromVariable(const std::string& name) const;
      
      // This function returns a string that shows the DataItem 
      // for this DataType.
      virtual std::string getDataItemString() const;

      // This function returns a string that shows the DataItem taht is used
      // at initialization for this DataType.
      virtual std::string getInitializationDataItemString() const;

      // This function returns a string that shows the DataItem 
      // for this DataType, if this dataType is in an array.     
      virtual std::string getArrayDataItemString() const;

      // This function returns if this dataType can be initialized using
      // a dataItem.
      virtual bool isLegitimateDataItem() const;

      // This function returns the code for initializing this dataType
      // if it is in an array.
      virtual std::string getArrayInitializerString(
	 const std::string& name,
	 const std::string& arrayName,
	 int level) const;

      // This function returns the code for initializing this dataType
      // if it is in an array. It is used for custom DataItems int and float.
      std::string getCustomArrayInitializerString(
	 const std::string& name,
	 const std::string& arrayName,
	 int level,
	 const std::string& diTypeName,
	 const std::string& dataTypeName) const;

      // This function sets the tab that is used in indentation using 
      // the level value.
      void setTabWithLevel(std::string& tab, int level) const;

      // This function returns if the pointer of the specific dataType
      // is meant to be owned by the class.
      virtual bool shouldBeOwned() const;

      // This function checks the if this dataItem has to be copied.
      virtual bool anythingToCopy();
     
      // This function returns a code string that is used by publisher's to 
      // create services.
      virtual std::string getServiceString(const std::string& tab) const;
      virtual std::string getServiceString(const std::string& tab, MachineType mach_type) const;

      
      // This function returns a code string that is used by publisher's to 
      // create optional services.
      virtual std::string getOptionalServiceString(
	 const std::string& tab) const;

      // This function returns code for the name of the service.
      virtual std::string getServiceNameString(const std::string& tab,
	    MachineType mach_type=MachineType::CPU
	    ) const;

      // This function returns code for the description of the service.
      virtual std::string getServiceDescriptionString(
	 const std::string& tab,
	 MachineType mach_type=MachineType::CPU
	 ) const;

      // This function returns code for the name of the service.
      std::string getOptionalServiceNameString(
	 const std::string& tab) const;

      // This function returns code for the description of the service.
      std::string getOptionalServiceDescriptionString(
	 const std::string& tab) const;

      // This function returns code for setting up the ServiceDescriptor.
      virtual std::string getServiceDescriptorString(
	 const std::string& tab) const;

      // This function returns code for setting up the ServiceDescriptor.
      virtual std::string getOptionalServiceDescriptorString(
	 const std::string& tab) const;

      // This function will add this dataType to a Class as a proxy attribute.
      virtual void addProxyAttribute(std::auto_ptr<Class>& instance) const;

      // This function will add a method to the given class to send this
      // dataType in a distributed environment.
      void addSenderMethod(Class& instance) const;
      
      // This function returns the sender method name for this datatype
      std::string getSenderMethodName() const;

      // This vector is used if an attribute is used to to something, 
      // e.g., input mapping instead of the dataType.
      const std::vector<std::string>& getSubAttributePath() const {
	 return _subAttributePath;
      }
      
      void setSubAttributePath(std::vector<std::string> path) {
	 _subAttributePath = path;
      }

      // This function sets the characteristics of the array container.
      virtual void setArrayCharacteristics(
	 unsigned blockSize, unsigned incrementSize);

   protected:
      // This function returns the code for checking a DataItems type.
      // If it is different than this DataType, an exception is thrown 
      // in the generated code.
      std::string getArgumentCheckerString(const std::string& name,
					   const std::string& diArg,
					   int level) const;

      // This function returns generated code; the returned value is
      // the function that is needed to call on the DataItem that 
      // returns the data of the DataItem.
      virtual std::string getDataItemFunctionString() const;

      std::string getNotLegitimateDataItemString(
	 const std::string& tab) const;

      bool _pointer;
      
   private:
      std::string getServiceInfoString(
	 const std::string& tab, const std::string& info,
	 MachineType mach_type = MachineType::CPU) const;

      std::string getOptionalServiceInfoString(
	 const std::string& tab, const std::string& info) const;

      bool _derived;
      bool _shared;
      std::string _name;
      std::string _name_gpu;
      std::string _name_gpu_raw;
      std::string _comment;
      std::vector<std::string> _subAttributePath;
};

#endif // DataType_H
