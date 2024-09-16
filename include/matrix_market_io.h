#ifndef MATRIX_MARKET_IO_H
#define MATRIX_MARKET_IO_H

#include "common.h"

namespace symmetria {
namespace io {

template <typename IT, typename DT>
CooTriples<IT, DT> * parse_mm_lines(int num_bytes, 
                                    int my_offset, 
                                    char * buf, 
                                    MPI_File& file_handle)
{
    using CooTripleVec = std::vector<std::tuple<IT, IT, DT>>;
    CooTripleVec tuples;

    /* We always want to advance to the end of the next line and copy that line into the buffer.
     * This is because, unless I'm rank 0, I always start parsing after the end of the first line I start on,
     * meaning the previous process has to read that line.
     */
    char * buf2 = new char[num_bytes];
    MPI_File_read_at(file_handle, my_offset+num_bytes, buf2, num_bytes, MPI_CHAR, MPI_STATUS_IGNORE);

    std::string buf2_str(buf2, num_bytes);
    size_t newline_idx = buf2_str.find('\n'); //index of next newline

    strncpy(buf + num_bytes, buf2, newline_idx+1); //copy up to next newline into buf

    num_bytes += (newline_idx + 1);

    delete[] buf2;

    //buf[num_bytes - 1] = '\0'; 

#ifdef DEBUG_DIST_SPMAT
    logptr->OFS()<<"Partition of file after line filling"<<std::endl;
    logptr->OFS()<<std::string(buf, num_bytes)<<std::endl;
#endif

    char * curr = buf;

    /* Advance until we find a newline unless I'm rank 0, which always starts at the start of a line */
    while (*curr!='\n' && my_pe!=0) {
        if (*curr=='\0') {
            ERROR("Found null terminator when parsing mm file for some reason");
        }
        curr++; 
    }

    if (my_pe != 0)
        curr++; //advance one more byte, should be at start of line
    
    std::string buf_str(curr);
    

#ifdef DEBUG_DIST_SPMAT
    logptr->OFS()<<"Parsing lines"<<std::endl;
#endif

    // Find end of this line
    size_t pos = 0;
    size_t next_eol = buf_str.find('\n', pos);
    size_t prev_eol = next_eol;
    size_t line_len = next_eol + 1;
    while ( next_eol != std::string::npos) {
        
        // Copy line into std::string 
        std::string line(curr+pos, line_len);
#ifdef DEBUG_DIST_SPMAT
//            logptr->OFS()<<line;
#endif
        std::istringstream iss(line);

        // Make the tuple
        IT row; IT col; DT val;
        iss>>row>>col>>val;
        tuples.emplace_back(row-1,col-1,val); //mm files are 1 indexed TODO: Always?
#ifdef DEBUG_DIST_SPMAT
        //logptr->OFS()<<row<<","<<col<<","<<val<<std::endl;
#endif

        // Advance to start of next line and find the end of the next line
        pos = (next_eol + 1);
        prev_eol = next_eol;
        next_eol = buf_str.find('\n', pos);
        line_len = (next_eol - prev_eol);
#ifdef DEBUG_DIST_SPMAT
//           logptr->OFS()<<"pos "<<pos<<", next_eol "<<next_eol<<", line_len "<<line_len<<std::endl;
#endif
    }
    return new CooTriples<IT,DT>(tuples);
}


template <typename IT, typename DT, typename Mat>
CooTriples<IT, DT> * distribute_tuples(CooTriples<IT, DT> * tuples, Mat& A)
{
    using Triple = std::tuple<IT, IT, DT>;
    using CooTripleVec = std::vector<Triple>;

#ifdef DEBUG_DIST_SPMAT
    std::cout<<"Distributing tuples"<<std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    /* Setup send buffers */
    std::vector<CooTripleVec> send_tuples(A.proc_map->get_n_procs());
    std::vector<int> send_sizes(A.proc_map->get_n_procs());
    std::vector<int> send_displs(A.proc_map->get_n_procs());

    for (auto& tuple : tuples->get_triples()) {
        /* Map tuple to correct process in cube */
        int target = A.map_triple(tuple);

#ifdef DEBUG_DIST_SPMAT
        //logptr->OFS()<<target<<std::endl;
#endif

        send_tuples[target].push_back(tuple);
        send_sizes[target]++;
    }

#ifdef DEBUG_DIST_SPMAT
    logptr->print_vec(send_sizes, "Send sizes");
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    CooTripleVec * send_buf = new CooTripleVec(
                                                 std::reduce(send_sizes.begin(),
                                                 send_sizes.end(),
                                                 0));

    for (int i=1; i<send_sizes.size(); i++) {
        send_displs[i] = send_displs[i-1] + send_sizes[i-1];
    }

#ifdef DEBUG_DIST_SPMAT
    logptr->print_vec(send_displs, "Send displs");
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    for (int i=0; i<send_tuples.size(); i++) {
        std::copy(send_tuples[i].begin(), send_tuples[i].end(),
                    send_buf->begin() + send_displs[i]);
    }

#ifdef DEBUG_DIST_SPMAT
    std::cout<<"Setup send buffers"<<std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    std::vector<int> recv_sizes(A.proc_map->get_n_procs());
    std::vector<int> recv_displs(A.proc_map->get_n_procs());
    MPI_Alltoall((void*)send_sizes.data(), 1, MPI_INT,
                    (void*)recv_sizes.data(), 1, MPI_INT,
                    MPI_COMM_WORLD);

    for (int i=1; i<recv_displs.size(); i++) {
        recv_displs[i] = recv_displs[i-1] + recv_sizes[i-1];
    }

#ifdef DEBUG_DIST_SPMAT
    std::cout<<"Did setup alltoall"<<std::endl;
    logptr->print_vec(recv_sizes, "Recv sizes");
    //logptr->print_tuple_vec(*(send_buf), "Send buf");
    MPI_Barrier(MPI_COMM_WORLD);
#endif


    CooTripleVec * recv_tuples = new CooTripleVec(
                                                    std::reduce(recv_sizes.begin(),
                                                    recv_sizes.end(),
                                                    0));
    MPI_Alltoallv(send_buf->data(), send_sizes.data(), send_displs.data(),
                    MPIType<Triple>(),
                    recv_tuples->data(), recv_sizes.data(), recv_displs.data(),
                    MPIType<Triple>(),
                    MPI_COMM_WORLD);
#ifdef DEBUG_DIST_SPMAT
    std::cout<<"Did alltoallv"<<std::endl;
    logptr->print_tuple_vec(*(recv_tuples), "Local tuples final", "End local tuples final");
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    delete send_buf;

    return new CooTriples<IT, DT>(*recv_tuples);
}


template <typename IT, typename DT, typename Mat>
void read_mm(const char * path, Mat& A)
{
    using CooTripleVec = std::vector<std::tuple<IT, IT, DT>>;

    CooTripleVec tuples;

    /* First, get nnz, dim info, and header offset */
    IT * send_buf = new IT[4];
    send_buf[3] = IT(0);

    if (my_pe==0) {

        std::string line;	

        std::ifstream mm_file(path);

        while (std::getline(mm_file, line)) {

            std::cout<<line<<std::endl;

            send_buf[3] += (strlen(line.c_str()) + 1);

            // Skip header
            if (line.find('%')!=std::string::npos) continue;
            
            std::istringstream iss(line);

            // First line after header is rows, cols, nnz
            iss>>send_buf[0]>>send_buf[1]>>send_buf[2];
            break;
                
        }

        mm_file.close();
    }

    MPI_Bcast(send_buf, 4, MPIType<IT>(), 0, MPI_COMM_WORLD);

    A.set_rows(send_buf[0]);
    A.set_cols(send_buf[1]);
    A.set_nnz(send_buf[2]);

    A.set_loc_dims();

    MPI_Offset header_offset = send_buf[3];

    delete[] send_buf;

    /* Begin MPI IO */
    MPI_File file_handle;
    MPI_File_open(MPI_COMM_WORLD, path, MPI_MODE_RDONLY, MPI_INFO_NULL, &file_handle);

    /* Compute offset info */
    MPI_Offset total_bytes;
    MPI_File_get_size(file_handle, &total_bytes);

    if (my_pe==0)
        std::cout<<"total bytes: "<<total_bytes<<std::endl;

    MPI_Offset my_offset = (header_offset) + (( ( total_bytes - header_offset ) / n_pes) * my_pe);


    int num_bytes = ((total_bytes - header_offset) / n_pes);  
    char *buf = new char[(size_t)(num_bytes*1.5 + 1)];//*1.5 ensures we have enough space to read in edge lines
                                                      //
    MPI_File_read_at(file_handle, my_offset, buf, num_bytes, MPI_CHAR, MPI_STATUS_IGNORE);

    MPI_Barrier(MPI_COMM_WORLD);


#ifdef DEBUG_DIST_SPMAT
    logptr->OFS()<<"Partition of file"<<std::endl;
    logptr->OFS()<<std::string(buf, num_bytes)<<std::endl;
#endif

    /* Parse my lines */
    auto read_tuples = parse_mm_lines<IT, DT>(num_bytes, my_offset, buf, file_handle);
    delete[] buf;

    /* Distribute tuples according to matrix distribution */
    auto local_tuples = distribute_tuples<IT, DT>(read_tuples, A);

    /* Map global tuple indices to local indices */
    std::transform(local_tuples->begin(), local_tuples->end(), local_tuples->begin(),
        [&](auto& tuple) {return A.map_glob_to_local(tuple);});

    /* Set local csr arrays */
    A.set_from_coo(local_tuples);

#ifdef DEBUG_DIST_SPMAT
    std::cout<<"Set csr ptrs"<<std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    delete local_tuples;

    MPI_File_close(&file_handle);
    MPI_Barrier(MPI_COMM_WORLD);

}


}//io
}//symmetria



#endif
