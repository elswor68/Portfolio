//Braden Elsworth
//Parallel and Concurrent Programming Project 1

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 16384

typedef struct
{
  int number;
  int count;
  int factors[64];
} NumberInfo;

void Build_number_info_type(NumberInfo *e, MPI_Datatype *number_info_mpi_t_p)
{

  //How much of each piece of the struct
  int array_of_blocklengths[3] = {1, 1, 64};

  //MPI type of each piece
  MPI_Datatype array_of_types[3] = {MPI_INT, MPI_INT, MPI_INT};

  //How fsr each piece is from the start of the struct
  MPI_Aint array_of_displacements[3];

  //Temp variable to hold each piece's memory address
  MPI_Aint number_addr, count_addr, factors_addr;
  
  // MPI gets the memory address of each piece
  MPI_Get_address(&e->number, &number_addr);
  MPI_Get_address(&e->count, &count_addr);
  MPI_Get_address(e->factors, &factors_addr);

  //0 distance away at the start
  //others are address minus the address of number
  array_of_displacements[0] = 0;
  array_of_displacements[1] = count_addr - number_addr;
  array_of_displacements[2] = factors_addr - number_addr;

  //Telling MPI how it can build the new type
  MPI_Type_create_struct(3, array_of_blocklengths, array_of_displacements, array_of_types, number_info_mpi_t_p);

  //Tel MPI the type is ready
  MPI_Type_commit(number_info_mpi_t_p);
  
}

//Build_number_info takes a number n and fills in one NumberInfo for it.
void Build_number_info(NumberInfo *ptr, int n)
{
  ptr->number = n;
  ptr->count = 0;

/*
  Try every number from 1 to n.
  If n divides evenly by i with a remainder of 0, then d is a factor.
*/
  for (int i = 1; i <= n; i++)
    {
      if (n % i == 0)
	{
	  ptr->factors[ptr->count] = i;
	  ptr->count++;
	}
    }
			    
}

int main(void)
{
  int comm_sz; //How many processes there are in total
  int my_rank; //Which process we're in
  MPI_Datatype number_info_mpi_t; 

  //Starting up MPI
  MPI_Init(NULL, NULL);

  //Retrieves # of processes in the communicator
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
  //Retrieves the rank for the process in the communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

  //Dividing each process equally
  int local_n = N / comm_sz;

  //Only get used by process 0 so leave it null for the others
  int *numbers = NULL;
  NumberInfo *results = NULL;

  //We're using malloc to reserve a # of bytes for local_n numbers and structs.
  int *local_numbers = malloc(local_n * sizeof(int));
  NumberInfo *local_infos = malloc(local_n * sizeof(NumberInfo));


  Build_number_info_type(&local_infos[0], &number_info_mpi_t);

  //Making sure we're in process 0 to make the random numbers
  if (my_rank == 0)
    {
      //Only process 0 needs the arrays
      numbers = malloc(N * sizeof(int));
      results = malloc(N * sizeof(NumberInfo));


      //srand uses a fixed seed for rand() and celebrates the sixers
      srand(76);

      //Rand gives a number between 0 and 4095 and shifts +1 to the right
      for (int i = 0; i < N; i++)
	{
	  numbers[i] = rand() % 4096 + 1;
	}
    }

  
/*
  We deal out the random numbers through scatter to each process.

  Arguments, in order:
         numbers        - where the full array lives (process 0 only)
         local_n        - how many numbers go to EACH process
         MPI_INT        - the type of those numbers
         local_numbers  - where each process puts its pile
         local_n        - how many numbers each process receives
         MPI_INT        - the type being received
         0              - the process doing the dealing (rank 0)
         MPI_COMM_WORLD - the group of all processes

*/
  MPI_Scatter(numbers, local_n, MPI_INT, local_numbers,
	      local_n, MPI_INT, 0, MPI_COMM_WORLD);


  //Each process will run this loop on their own group of numbers
  for (int i = 0; i < local_n; i++)
    {
      Build_number_info(&local_infos[i], local_numbers[i]);
    }


/*
  Now each process sends its local_n structs to process 0
  which will stack them in rank order.
*/
  MPI_Gather(local_infos, local_n, number_info_mpi_t,
	     results, local_n, number_info_mpi_t,
	     0, MPI_COMM_WORLD);

  //Only print if at process 0  
  if (my_rank == 0)
    {
      for (int i = 0; i < 10; i++)
	{
	  //pick a random position
	  int idx = rand() % N;
	  //point at the struct
	  NumberInfo *e = &results[idx];

	  printf("Index %d: number %d, factor count = %d\n factors:",
		 idx, e->number, e->count);

	  //Only print factors we filled in
	  for (int j = 0; j < e->count; j++)
	    {
	      printf(" %d", e->factors[j]);	     
	    }
	  printf("\n\n");
	}

      //Free memory process 0 reserved
      free(numbers);
      free(results);
    }

  //Free memory every process reserved
  free(local_numbers);
  free(local_infos);

  //Clean up and shut down MPI
  MPI_Type_free(&number_info_mpi_t);
  MPI_Finalize();

  return 0;
}
