#include "sptensor.h"
#include "ciss.h"
#include "util.h"

/*
This is used for MBLC
*/

/**
The private function
**/
static void p_rearrange(sptensor_t* tt, idx_t mode)
{ idx_t nnz = tt->nnz;  //mode1: i,j,k; mode2:j,i,k; mode3:k,i,j
  idx_t tmp0_length = tt->dims[mode-1];
  #ifdef CISS_DEBUG
  printf("tmp0_length %d\n", tmp0_length);
  printf("nnz %d and mode %d\n", nnz, mode);
  #endif
  idx_t * temparray0 =(idx_t*)malloc((tmp0_length+1) * sizeof(idx_t));
  idx_t * temparray1 =(idx_t*)malloc((tmp0_length+1) * sizeof(idx_t));
  memset(temparray0, 0, (tmp0_length+1) * sizeof(idx_t));
  memset(temparray1, 0, (tmp0_length+1) * sizeof(idx_t));
  idx_t sortnum = mode - 1;
  idx_t i;
  #ifdef CISS_DEBUG
  printf("begin rearrange\n");
  #endif
  if(mode == 1) {free(temparray0);free(temparray1);return;}
  else
  { if(mode == 2)
    {
     quicksort(tt->ind[1], tt->ind[2], tt->ind[0], tt->vals, 0, nnz-1);
     }
   else quicksort(tt->ind[2], tt->ind[0], tt->ind[1], tt->vals, 0, nnz-1);
   }
  for(i=0;i<nnz;i++)
  {
    temparray0[(tt->ind)[sortnum][i]-1]++;
    
  }
  
  idx_t count = 0;
  temparray1[count] = 0;
  count++;
  idx_t j;
  for(i=0;i<tmp0_length;i++)
  {
    if(temparray0[i]!=0) {temparray1[count] = temparray0[i]+temparray1[count-1];count++; 
    //printf("temparray1[%d] %d\n",count-1,temparray1[count-1]);
   }
  }
  free(temparray0);
  
  for(i=0;i<count-1;i++)
  {
    quicksort(tt->ind[sortnum],tt->ind[(sortnum+1)%(tt->nmodes)],tt->ind[(sortnum+2)%(tt->nmodes)],tt->vals,temparray1[i],temparray1[i+1]-1);
    }
  free(temparray1); 
  } 



/**
 * @brief Alloc the tensor in MBLC
 * @param newtensor the original tensor in COO
 * @param blockref the array for guidance the partition
*/ 
ciss_t* ciss_alloc(
    sptensor_t * newtensor,
    idx_t mode
)
{
    //rearrange
    //p_rearrange(newtensor, mode);
    //intialization
    mode--;
    idx_t m, counter = 1, lastelement;
    static ciss_t* resultensor = NULL;
    resultensor = (ciss_t*)malloc(sizeof(ciss_t));
    resultensor->nmodes = newtensor->nmodes;
    resultensor->nnz = newtensor->nnz;
    resultensor->dims = (idx_t*)malloc(resultensor->nmodes * sizeof(idx_t));
    for(m = 0; m < resultensor->nmodes; m++)
    {
        resultensor->dims[m] = newtensor->dims[m];
    }

    //create the directory
    idx_t* tmp_directory = (idx_t*)malloc(sizeof(idx_t) * newtensor->nnz);
    idx_t* tmp_dcounter = (idx_t*)malloc(sizeof(idx_t) * newtensor->nnz);
    memset(tmp_dcounter, 0, newtensor->nnz * sizeof(idx_t));
    lastelement = newtensor->ind[mode][0];
    tmp_directory[0] = lastelement;//第一个元素的第一维坐标
    #ifdef CISS_DEBUG
    printf("lastelement %d\n", lastelement);
    #endif
    tmp_dcounter[0] = 0;
    for(m = 1; m < resultensor->nnz; m++)
    {
        if(lastelement != newtensor->ind[mode][m])
        {
            tmp_directory[counter] = newtensor->ind[mode][m]; 
            counter++;
            tmp_dcounter[counter] += tmp_dcounter[counter-1];
            lastelement = newtensor->ind[mode][m];           
        }
        tmp_dcounter[counter]++;
    }
    
    resultensor->directory = (idx_t*)malloc(counter * sizeof(idx_t));
    resultensor->dcounter = (idx_t*)malloc((counter + 1)* sizeof(idx_t));
    resultensor->dcounter[counter] = resultensor->nnz - 1;
    memcpy(resultensor->directory, tmp_directory, counter * sizeof(idx_t));
    memcpy(resultensor->dcounter, tmp_dcounter, counter * sizeof(idx_t));
    resultensor->dlength = counter;
    
    #ifdef CISS_DEBUG
    printf("finish directory\n");
    #endif
    //create the entries
    idx_t bitmap, tilesize = (resultensor->nnz)/DEFAULT_T_TILE_LENGTH + 1;
    idx_t nresidual = 0, nposition = 0, nivalue = 0, nindice_s, nindice_l = 0;
    idx_t lcounter, tmpi, tmpj, tmpk, tmpv; //the local counter inside each tile
    resultensor->entries = (double*)malloc((tilesize + resultensor->nnz + 1)
    *DEFAULT_T_TILE_WIDTH*sizeof(double));
    resultensor->size = tilesize + resultensor->nnz + 1;
    double* entry = resultensor->entries;
    double* lentry = entry; //the local entry

    //the position of indices in mode-(mode) will multiply (-1) and if the indices, if other indices == -1 then indicates the end
    for( m = 0 ; m < resultensor->nnz + 1 ;)
    {
        //initialize for the tile
        bitmap = 1;
        nindice_s = nindice_l;
        nivalue = tmp_directory[nindice_s];
        lcounter = 1;
        lentry = entry+nposition;
        lentry[0] = -1 * (double)nindice_s;

        //check and add 一片
        while(lcounter <= DEFAULT_T_TILE_LENGTH)
        {
            if(m == resultensor->nnz)   //now the end
            {
                lentry[lcounter*DEFAULT_T_TILE_WIDTH] = -1;
                lentry[lcounter*DEFAULT_T_TILE_WIDTH + 1] = -1;
                lentry[lcounter*DEFAULT_T_TILE_WIDTH + 2] = -1;
                m++;
                break;
            } 
            else
            {
                tmpi = newtensor->ind[mode][m];
                tmpj = newtensor->ind[(mode+1)%(newtensor->nmodes)][m];            
                tmpk = newtensor->ind[(mode+2)%(newtensor->nmodes)][m];
                tmpv = newtensor->vals[m];
                if(tmpi == nivalue)
                {
                    bitmap = (bitmap<<1) + 1;                    
                }
                else //when the i change
                {
                    bitmap = (bitmap<<1);
                    //#ifdef CISS_DEBUG
                    //printf("now i changes\n");
                    //printf("the bitmap is %d\n", bitmap);
                    //#endif
                    nivalue = tmpi;
                    nindice_l++;
                    lentry[1] = -1 * (double)nindice_l;
                }
                //store the rest
                lentry[lcounter*DEFAULT_T_TILE_WIDTH] = tmpj;
                lentry[lcounter*DEFAULT_T_TILE_WIDTH + 1] = tmpk;
                lentry[lcounter*DEFAULT_T_TILE_WIDTH + 2] = tmpv;
                lentry[2] = bitmap;               
            }        
        
            //update the position
            m++;
            lcounter++;
            #ifdef DEBUG
            printf("finish %d\n",m);
            #endif
        }
        nposition += (DEFAULT_T_TILE_LENGTH + 1) * DEFAULT_T_TILE_WIDTH;
    }

    free(tmp_directory);
    #ifdef CISS_DEBUG
    printf("mode %d finish\n", mode);
    #endif
    return resultensor;
}

/**
 * @brief Alloc the tensor in MBLC
 * @param newtensor the original tensor in COO
 * @param blockref the array for guidance the partition
*/ 
blcotensor* blco_alloc(
    sptensor_t * newtensor,
    idx_t mode
){
    
    return NULL;
}

/**
 * @brief The function for copy between MBLC
*/ 
ciss_t* ciss_copy(
        ciss_t * oldtensor
)
{
    static ciss_t* newtensor = NULL;
    newtensor = (ciss_t*)malloc(sizeof(ciss_t));
    newtensor->nmodes = oldtensor->nmodes;
    newtensor->nnz = oldtensor->nnz;
    newtensor->dlength = oldtensor->dlength;
    newtensor->size = oldtensor->size;
    idx_t m;
    for(m=0;m<oldtensor->nmodes;m++)
    {
        newtensor->dims[m] = oldtensor->dims[m];
    }
    memcpy(newtensor->directory, oldtensor->directory, oldtensor->dlength*sizeof(idx_t));
    memcpy(newtensor->dcounter, oldtensor->dcounter, (oldtensor->dlength + 1)*sizeof(idx_t));
    memcpy(newtensor->entries, oldtensor->entries, oldtensor->size*sizeof(double));
    return newtensor;
}

/**
 * @brief The function for display the MBLC
*/ 
void ciss_display(ciss_t* newtensor)
{
    fprintf(stdout, "The MBLC tensor\n");
    fprintf(stdout, "The nmode is %d\n", newtensor->nmodes);
    fprintf(stdout, "The number of nnz is %d\n", newtensor->nnz);
    fprintf(stdout, "The length of directory is %d\n", newtensor->dlength);
    fprintf(stdout, "The length of entries is %d\n", newtensor->size);
    idx_t m;
    for(m=0;m<newtensor->nmodes;m++)
    {
        fprintf(stdout, "The length of %dth dimension is %d\n", m, newtensor->dims[m]);
    }
    fprintf(stdout, "Now is the directory \n");
    for(m=0;m<newtensor->dlength;m++)
    {
        fprintf(stdout, "Directory[%d] = %d\n", m, newtensor->directory[m]);
        fprintf(stdout, "Dcounter[%d] = %d\n", m, newtensor->dcounter[m]);
    }
    fprintf(stdout, "Dcounter[%d] = %d\n", m, newtensor->dcounter[m]);
    fprintf(stdout, "Now is the entries \n");
    for(m=0;m<newtensor->size;m++)
    {
        fprintf(stdout, "Entry[%d] = (%f, %f, %f)\n", m, newtensor->entries[m*DEFAULT_T_TILE_WIDTH],newtensor->entries[m*DEFAULT_T_TILE_WIDTH+1],newtensor->entries[m*DEFAULT_T_TILE_WIDTH+2]);
    }
    fprintf(stdout, "Now the display finishes\n");
}

/**
 *@brief The function for free the MBLC 
*/ 
void ciss_free(ciss_t* newtensor)
{
    free(newtensor->directory);
    free(newtensor->dims);
    free(newtensor->entries);
    free(newtensor);
}


