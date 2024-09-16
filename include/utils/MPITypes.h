#ifndef MPI_TYPES_HPP
#define MPI_TYPES_HPP

#include <mpi.h>
#include <map>

namespace symmetria{

struct type_info_compare
{
  bool operator()(std::type_info const* lhs, std::type_info const* rhs) const
  {
    return lhs->before(*rhs);
  }
};


class MPIDataTypeCache
{
private:
  typedef std::map<std::type_info const*,MPI_Datatype,type_info_compare> stored_map_type;
  stored_map_type map;

public:
  void clear()
  {
	int is_finalized=0;
	MPI_Finalized(&is_finalized);
	if (! is_finalized ) 	// do not free after call to MPI_FInalize
	{
		// ignore errors in the destructor
		for (stored_map_type::iterator it=map.begin(); it != map.end(); ++it)
		{
			MPI_Type_free(&(it->second));
		}
	}
  }
  ~MPIDataTypeCache()
  {
    	clear();
  }
  MPI_Datatype get(const std::type_info* t)
  {
      	stored_map_type::iterator pos = map.find(t);
      	if (pos != map.end())
          	return pos->second;
      	else
        	return MPI_DATATYPE_NULL;
  }

  void set(const std::type_info* t, MPI_Datatype datatype)
  {
     	 map[t] = datatype;
  }
};


/**
  * C++ type to MPIType conversion is done through functions returning the mpi types
  * The templated function is explicitly instantiated for every C++ type 
  * that has a correspoinding MPI type. For all others, a data type is created
  * assuming it's some sort of struct. Each created data type is committed only once
  **/

MPIDataTypeCache mpidtc;	// global variable

template <typename T> 
MPI_Datatype MPIType ( void )
{
	std::type_info const* t = &typeid(T);
	MPI_Datatype datatype = mpidtc.get(t);

	if (datatype == MPI_DATATYPE_NULL) 
	{
		MPI_Type_contiguous(sizeof(T), MPI_CHAR, &datatype );
		MPI_Type_commit(&datatype);
		int myrank;
		MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
		mpidtc.set(t, datatype);
	}
   	return datatype;
};

template<> MPI_Datatype MPIType< signed char >( void )
{
	return MPI_CHAR;
}; 
template<> MPI_Datatype MPIType< unsigned char >( void )
{
	return MPI_UNSIGNED_CHAR;
}; 
template<> MPI_Datatype MPIType< signed short int >( void )
{
	return MPI_SHORT;
}; 
template<> MPI_Datatype MPIType< unsigned short int >( void )
{
	return MPI_UNSIGNED_SHORT;
}; 
template<> MPI_Datatype MPIType< int32_t >( void )
{
	return MPI_INT;
};  
template<> MPI_Datatype MPIType< uint32_t >( void )
{
	return MPI_UNSIGNED;
};
template<> MPI_Datatype MPIType<int64_t>(void)
{
	return MPI_LONG_LONG;
};
template<> MPI_Datatype MPIType< uint64_t>(void)
{
	return MPI_UNSIGNED_LONG_LONG;
};
template<> MPI_Datatype MPIType< float >( void )
{
	return MPI_FLOAT;
}; 
template<> MPI_Datatype MPIType< double >( void )
{
	return MPI_DOUBLE;
}; 
template<> MPI_Datatype MPIType< long double >( void )
{
	return MPI_LONG_DOUBLE;
}; 
template<> MPI_Datatype MPIType< bool >( void )
{
	return MPI_BYTE;  // usually  #define MPI_BOOL MPI_BYTE anyway
};

}


#endif
